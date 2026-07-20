//! Web search and code search tools.

use anyhow::{Context, Result};
use std::io::{BufRead, Read};
#[cfg(unix)]
use std::os::unix::process::CommandExt;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use super::shell::kill_process_group;
use super::web::{read_error_body, web_client};

/// Wall-clock ceiling for symbol_search / grep_search.
const SEARCH_TIMEOUT_SECS: u64 = 30;

// ─── Web search providers ─────────────────────────────────────────────────────

fn search_brave(
    query: &str,
    api_key: &str,
    kind: Option<&str>,
    freshness: Option<&str>,
) -> Result<String> {
    let endpoint = if kind == Some("news") {
        "https://api.search.brave.com/res/v1/news/search"
    } else {
        "https://api.search.brave.com/res/v1/web/search"
    };
    let mut req = web_client()
        .get(endpoint)
        .query(&[("q", query), ("count", "10"), ("extra_snippets", "true")])
        .header("X-Subscription-Token", api_key)
        .header("Accept", "application/json");
    if let Some(f) = freshness {
        req = req.query(&[("freshness", f)]);
    }
    let resp = req.send().context("brave search request failed")?;
    if !resp.status().is_success() {
        let status = resp.status();
        let body = read_error_body(resp);
        anyhow::bail!(
            "brave search returned {}: {}",
            status,
            body.chars().take(300).collect::<String>()
        );
    }
    let json: serde_json::Value = resp.json().context("parse brave response")?;
    let results = if kind == Some("news") {
        json["results"]
            .as_array()
            .map(|a| a.as_slice())
            .unwrap_or(&[])
    } else {
        json["web"]["results"]
            .as_array()
            .map(|a| a.as_slice())
            .unwrap_or(&[])
    };
    if results.is_empty() {
        return Ok("No results found.".into());
    }
    let mut out = String::new();
    for r in results.iter().take(10) {
        let title = r["title"].as_str().unwrap_or("(no title)");
        let url = r["url"].as_str().unwrap_or("");
        let desc = r["description"].as_str().unwrap_or("");
        out.push_str(&format!("- **{}** <{}>\n  {}\n", title, url, desc));
        if let Some(extras) = r["extra_snippets"].as_array() {
            for snippet in extras.iter().take(3) {
                if let Some(s) = snippet.as_str() {
                    out.push_str(&format!("  > {}\n", s));
                }
            }
        }
    }
    Ok(out)
}

fn search_pipellm(query: &str, api_key: &str, kind: Option<&str>) -> Result<String> {
    let path = match kind {
        Some("news") => "v1/websearch/search-news",
        Some("deep") => "v1/websearch/search",
        _ => "v1/websearch/simple-search",
    };
    let domains = ["https://api.pipellm.ai", "https://api.pipellm.com"];
    let mut last_err = String::new();
    for base in &domains {
        let url = format!("{}/{}", base, path);
        let resp = match web_client()
            .get(&url)
            .query(&[("q", query)])
            .bearer_auth(api_key)
            .send()
        {
            Ok(r) => r,
            Err(e) => {
                last_err = e.to_string();
                continue;
            }
        };
        if !resp.status().is_success() {
            let status = resp.status();
            let body = read_error_body(resp);
            last_err = format!(
                "{} from {}: {}",
                status,
                base,
                body.chars().take(300).collect::<String>()
            );
            continue;
        }
        let json: serde_json::Value = resp.json().context("parse pipellm response")?;
        let results = json["organic"]
            .as_array()
            .or_else(|| json["data"]["organic"].as_array())
            .map(|a| a.as_slice())
            .unwrap_or(&[]);
        if results.is_empty() {
            return Ok("No results found.".into());
        }
        let mut out = String::new();
        for r in results.iter().take(10) {
            let title = r["title"].as_str().unwrap_or("(no title)");
            let url = r["link"]
                .as_str()
                .or_else(|| r["url"].as_str())
                .unwrap_or("");
            let snippet = r["snippet"]
                .as_str()
                .or_else(|| r["content"].as_str())
                .unwrap_or("");
            out.push_str(&format!("- **{}** <{}>\n  {}\n", title, url, snippet));
        }
        return Ok(out);
    }
    anyhow::bail!("pipellm search failed: {}", last_err)
}

fn search_tavily(
    query: &str,
    api_key: &str,
    kind: Option<&str>,
    freshness: Option<&str>,
    search_depth: Option<&str>,
) -> Result<String> {
    let mut body = serde_json::json!({
        "query": query,
        "max_results": 10,
        "include_answer": true
    });
    if let Some(k) = kind {
        if k == "news" || k == "finance" {
            body["topic"] = serde_json::json!(k);
        }
    }
    if let Some(d) = search_depth {
        body["search_depth"] = serde_json::json!(d);
    }
    if let Some(f) = freshness {
        let days: u32 = match f {
            "pd" => 1,
            "pw" => 7,
            "pm" => 31,
            "py" => 365,
            other => other.parse().unwrap_or(7),
        };
        body["days"] = serde_json::json!(days);
    }
    let resp = web_client()
        .post("https://api.tavily.com/search")
        .bearer_auth(api_key)
        .json(&body)
        .send()
        .context("tavily search request failed")?;
    if !resp.status().is_success() {
        let status = resp.status();
        let body = read_error_body(resp);
        anyhow::bail!(
            "tavily search returned {}: {}",
            status,
            body.chars().take(300).collect::<String>()
        );
    }
    let json: serde_json::Value = resp.json().context("parse tavily response")?;
    let results = json["results"]
        .as_array()
        .map(|a| a.as_slice())
        .unwrap_or(&[]);
    let mut out = String::new();
    if let Some(answer) = json["answer"].as_str() {
        if !answer.is_empty() {
            out.push_str(&format!("**Answer:** {}\n\n", answer));
        }
    }
    if results.is_empty() && out.is_empty() {
        return Ok("No results found.".into());
    }
    for r in results.iter().take(10) {
        let title = r["title"].as_str().unwrap_or("(no title)");
        let url = r["url"].as_str().unwrap_or("");
        let content = r["content"].as_str().unwrap_or("");
        out.push_str(&format!("- **{}** <{}>\n  {}\n", title, url, content));
    }
    Ok(out)
}

pub(super) fn exec_web_search(
    args: &serde_json::Value,
    config: &crate::ai_client::AssistantConfig,
) -> Result<String> {
    let query = args["query"].as_str().context("missing query")?;
    let provider = config
        .web_search_provider
        .as_deref()
        .context("web_search provider not configured")?;
    let api_key = config
        .web_search_api_key
        .as_deref()
        .context("web_search api key missing")?;
    let kind = args["kind"].as_str();
    let freshness = args["freshness"].as_str();
    let search_depth = args["search_depth"].as_str();
    match provider {
        "brave" => search_brave(query, api_key, kind, freshness),
        "pipellm" => search_pipellm(query, api_key, kind),
        "tavily" => search_tavily(query, api_key, kind, freshness, search_depth),
        _ => anyhow::bail!("unknown web_search provider: {}", provider),
    }
}

// ─── Code search ──────────────────────────────────────────────────────────────

fn escape_regex(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 8);
    for c in s.chars() {
        if matches!(
            c,
            '\\' | '.' | '+' | '*' | '?' | '(' | ')' | '|' | '[' | ']' | '{' | '}' | '^' | '$'
        ) {
            out.push('\\');
        }
        out.push(c);
    }
    out
}

pub(super) fn exec_symbol_search(
    query: &str,
    kind: &str,
    search_path: &str,
    glob_filter: Option<&str>,
    cwd: &str,
    cancel: &Arc<AtomicBool>,
) -> Result<String> {
    let abs_path = super::paths::resolve(search_path, cwd)?
        .to_string_lossy()
        .into_owned();

    let patterns: Vec<String> = match kind {
        "function" => vec![
            format!(r"(fn|function|def|func)\s+{}", escape_regex(query)),
            format!(
                r"(const|let|var)\s+{}\s*=\s*(async\s+)?\(",
                escape_regex(query)
            ),
        ],
        "type" => vec![format!(
            r"(type|struct|enum|interface|typedef)\s+{}",
            escape_regex(query)
        )],
        "class" => vec![format!(r"(class|struct)\s+{}", escape_regex(query))],
        "method" => vec![
            format!(r"(fn|def|func|function)\s+{}", escape_regex(query)),
            format!(r"\.{}\s*=\s*function", escape_regex(query)),
        ],
        _ => vec![
            format!(r"(fn|function|def|func)\s+{}", escape_regex(query)),
            format!(
                r"(const|let|var)\s+{}\s*=\s*(async\s+)?\(",
                escape_regex(query)
            ),
            format!(
                r"(type|struct|enum|interface|class|trait|typedef)\s+{}",
                escape_regex(query)
            ),
            format!(r"(pub\s+)?(mod|module)\s+{}", escape_regex(query)),
        ],
    };
    let combined = patterns.join("|");

    static HAS_RG: OnceLock<bool> = OnceLock::new();
    let rg = *HAS_RG.get_or_init(|| {
        std::process::Command::new("rg")
            .arg("--version")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .is_ok()
    });

    let mut cmd = if rg {
        let mut c = std::process::Command::new("rg");
        c.arg("--line-number")
            .arg("--no-heading")
            .arg("--color=never")
            .arg("--max-count=50");
        if let Some(g) = glob_filter {
            c.arg("--glob").arg(g);
        }
        c.arg(&combined).arg(&abs_path);
        c
    } else {
        let mut c = std::process::Command::new("grep");
        c.arg("-rn").arg("--color=never").arg("-E");
        if let Some(g) = glob_filter {
            c.arg("--include").arg(g);
        }
        c.arg(&combined).arg(&abs_path);
        c
    };

    cmd.stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null());
    #[cfg(unix)]
    cmd.process_group(0);
    let mut child = cmd.spawn().context("symbol_search exec failed")?;

    let stdout_pipe = child
        .stdout
        .take()
        .ok_or_else(|| anyhow::anyhow!("symbol_search stdout missing"))?;
    let collected = Arc::new(Mutex::new(Vec::<u8>::new()));
    let collected_clone = collected.clone();
    let reader_thread = crate::thread_util::spawn_with_pool(move || {
        let mut r = stdout_pipe;
        let mut buf = [0u8; 8192];
        loop {
            match r.read(&mut buf) {
                Ok(0) | Err(_) => break,
                Ok(n) => {
                    if let Ok(mut g) = collected_clone.lock() {
                        g.extend_from_slice(&buf[..n]);
                    }
                }
            }
        }
    });

    let start = Instant::now();
    let timeout = Duration::from_secs(SEARCH_TIMEOUT_SECS);
    let mut timed_out = false;
    loop {
        if cancel.load(Ordering::Relaxed) {
            kill_process_group(&child);
            child.wait().ok();
            let _ = reader_thread.join();
            anyhow::bail!("symbol_search canceled");
        }
        if start.elapsed() >= timeout {
            kill_process_group(&child);
            timed_out = true;
            break;
        }
        if let Ok(Some(_)) = child.try_wait() {
            break;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    child.wait().ok();
    let _ = reader_thread.join();

    let raw = collected.lock().map(|g| g.clone()).unwrap_or_default();
    let text = String::from_utf8_lossy(&raw);

    if text.trim().is_empty() {
        if timed_out {
            return Ok(format!(
                "symbol_search timed out after {}s with no results for '{}'.",
                SEARCH_TIMEOUT_SECS, query
            ));
        }
        return Ok(format!("No symbol definitions found for '{}'.", query));
    }

    let mut lines: Vec<&str> = text.lines().take(100).collect();
    lines.sort_by(|a, b| {
        let a_kw = a.contains("fn ")
            || a.contains("function ")
            || a.contains("def ")
            || a.contains("struct ")
            || a.contains("class ")
            || a.contains("type ")
            || a.contains("enum ")
            || a.contains("trait ")
            || a.contains("interface ");
        let b_kw = b.contains("fn ")
            || b.contains("function ")
            || b.contains("def ")
            || b.contains("struct ")
            || b.contains("class ")
            || b.contains("type ")
            || b.contains("enum ")
            || b.contains("trait ")
            || b.contains("interface ");
        b_kw.cmp(&a_kw)
    });

    let mut out = lines.join("\n");
    if timed_out {
        out.push_str(&format!(
            "\n[... timed out after {}s, results may be partial]",
            SEARCH_TIMEOUT_SECS
        ));
    }
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn exec_grep_search(
    pattern: &str,
    search_path: &str,
    glob_filter: Option<&str>,
    context_lines: usize,
    case_insensitive: bool,
    max_results: usize,
    cwd: &str,
    cancel: &Arc<AtomicBool>,
) -> Result<String> {
    static HAS_RG: OnceLock<bool> = OnceLock::new();
    let rg = *HAS_RG.get_or_init(|| {
        std::process::Command::new("rg")
            .arg("--version")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .is_ok()
    });
    let abs_path = super::paths::resolve(search_path, cwd)?
        .to_string_lossy()
        .into_owned();

    let mut cmd = if rg {
        let mut c = std::process::Command::new("rg");
        c.arg("--line-number")
            .arg("--no-heading")
            .arg("--color=never")
            .arg(format!("--context={}", context_lines))
            .arg(format!("--max-count={}", max_results));
        if case_insensitive {
            c.arg("--ignore-case");
        }
        if let Some(g) = glob_filter {
            c.arg("--glob").arg(g);
        }
        c.arg(pattern).arg(&abs_path);
        c
    } else {
        let mut c = std::process::Command::new("grep");
        c.arg("-rn")
            .arg(format!("-C{}", context_lines))
            .arg("--color=never");
        if case_insensitive {
            c.arg("-i");
        }
        if let Some(g) = glob_filter {
            c.arg("--include").arg(g);
        }
        c.arg(pattern).arg(&abs_path);
        c
    };

    cmd.stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());
    #[cfg(unix)]
    cmd.process_group(0);
    let mut child = cmd.spawn().context("grep_search exec failed")?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| anyhow::anyhow!("grep stdout missing"))?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| anyhow::anyhow!("grep stderr missing"))?;

    let stderr_buf: Arc<Mutex<Vec<u8>>> = Arc::new(Mutex::new(Vec::with_capacity(512)));
    let stderr_buf_clone = stderr_buf.clone();
    let stderr_handle = crate::thread_util::spawn_with_pool(move || {
        let mut err = stderr;
        let mut chunk = [0u8; 512];
        loop {
            match err.read(&mut chunk) {
                Ok(0) | Err(_) => break,
                Ok(n) => {
                    if let Ok(mut g) = stderr_buf_clone.lock() {
                        let remaining = 512usize.saturating_sub(g.len());
                        if remaining > 0 {
                            g.extend_from_slice(&chunk[..remaining.min(n)]);
                        }
                    }
                }
            }
        }
    });

    let result_lines: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let match_count: Arc<AtomicUsize> = Arc::new(AtomicUsize::new(0));
    let truncated_flag = Arc::new(AtomicBool::new(false));

    let rl = result_lines.clone();
    let mc = match_count.clone();
    let tf = truncated_flag.clone();
    let max = max_results;
    let reader_handle = crate::thread_util::spawn_with_pool(move || {
        let reader = std::io::BufReader::new(stdout);
        for line_result in reader.lines() {
            let line = match line_result {
                Ok(l) => l,
                Err(_) => break,
            };
            if !line.starts_with("--") {
                if mc.load(Ordering::Relaxed) >= max {
                    tf.store(true, Ordering::Relaxed);
                    break;
                }
                mc.fetch_add(1, Ordering::Relaxed);
            }
            if let Ok(mut g) = rl.lock() {
                g.push(line);
            }
        }
    });

    let start = Instant::now();
    let timeout = Duration::from_secs(SEARCH_TIMEOUT_SECS);
    let mut timed_out = false;
    let mut canceled = false;
    loop {
        if cancel.load(Ordering::Relaxed) {
            kill_process_group(&child);
            canceled = true;
            break;
        }
        if start.elapsed() >= timeout {
            kill_process_group(&child);
            timed_out = true;
            break;
        }
        if let Ok(Some(_)) = child.try_wait() {
            break;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    child.wait().ok();
    let _ = reader_handle.join();
    let _ = stderr_handle.join();

    let truncated = truncated_flag.load(Ordering::Relaxed);
    let lines = result_lines.lock().map(|g| g.clone()).unwrap_or_default();

    if lines.is_empty() {
        if canceled {
            anyhow::bail!("grep_search canceled");
        }
        if timed_out {
            return Ok(format!(
                "grep_search timed out after {}s with no results.",
                SEARCH_TIMEOUT_SECS
            ));
        }
        let hint = stderr_buf
            .lock()
            .ok()
            .map(|g| {
                String::from_utf8_lossy(&g)
                    .trim()
                    .chars()
                    .take(200)
                    .collect::<String>()
            })
            .unwrap_or_default();
        if !hint.is_empty() {
            return Ok(format!("No matches. ({})", hint));
        }
        return Ok("No matches found.".into());
    }

    let mut out = lines.join("\n");
    if truncated {
        out.push_str(&format!("\n[... truncated at {} results]", max_results));
    }
    if timed_out {
        out.push_str(&format!(
            "\n[... timed out after {}s, results may be partial]",
            SEARCH_TIMEOUT_SECS
        ));
    }
    if canceled {
        out.push_str("\n[... canceled by user]");
    }
    Ok(out)
}
