use crate::api::error::{ApiError, ApiResult};
use crate::config::Config;
use crate::content::upload::MAX_UPLOAD_SIZE;
use crate::filesystem::{self, Directory};
use crate::model::enums::MimeType;
use crate::string::SmallString;
use reqwest::Client;
use reqwest::header::{CONTENT_TYPE, HeaderMap, HeaderValue, REFERER};
use std::path::{Path, PathBuf};
use std::str::FromStr;
use tokio::process::Command;
use url::Url;
use uuid::Uuid;

const FAKE_USER_AGENT: &str =
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:135.0) Gecko/20100101 Firefox/135.0";
const YTDLP_FORMAT_SELECTOR: &str =
    "best[ext=jpg]/best[ext=jpeg]/best[ext=png]/best[ext=gif]/best[ext=webp]/best[ext=bmp]/\
best[ext=avif]/best[ext=mp4]/best[ext=webm]/best[ext=mov]";

/// Attempts to download file at the specified `url`.
/// If successful, the file is saved in the temporary uploads directory
/// and a content token is returned.
pub async fn from_url(config: &Config, url: Url, allow_downloader: bool) -> ApiResult<String> {
    match download_direct(config, url.clone()).await {
        Ok(token) => Ok(token),
        Err(err) => {
            if allow_downloader && matches!(err, ApiError::UnsupportedContentType(_)) {
                return download_with_downloader(config, url).await;
            }
            Err(err)
        }
    }
}

async fn download_direct(config: &Config, url: Url) -> ApiResult<String> {
    let mut headers = HeaderMap::new();
    headers.insert(REFERER, HeaderValue::from_str(url.as_str())?);

    let client = Client::builder()
        .user_agent(FAKE_USER_AGENT)
        .default_headers(headers)
        .build()?;
    let response = client.get(url.clone()).send().await?;
    let response = response.error_for_status()?;

    let raw_content_type = response
        .headers()
        .get(CONTENT_TYPE)
        .and_then(|header_value| header_value.to_str().ok())
        .unwrap_or("");
    let content_type = normalize_content_type(raw_content_type);

    let mime_type = if let Ok(mime_type) = MimeType::from_str(&content_type) {
        mime_type
    } else if should_infer_mime_from_url(&content_type) {
        let label = if content_type.is_empty() {
            "unknown"
        } else {
            content_type.as_str()
        };
        mime_type_from_url(&url).ok_or_else(|| ApiError::UnsupportedContentType(SmallString::new(label)))?
    } else {
        let label = if content_type.is_empty() {
            "unknown"
        } else {
            content_type.as_str()
        };
        return Err(ApiError::UnsupportedContentType(SmallString::new(label)));
    };

    let bytes = response.bytes().await?;
    filesystem::save_uploaded_file(config, &bytes, mime_type).map_err(ApiError::from)
}

async fn download_with_downloader(config: &Config, url: Url) -> ApiResult<String> {
    let temp_dir = config.path(Directory::TemporaryUploads);
    std::fs::create_dir_all(&temp_dir)?;

    let token_prefix = Uuid::new_v4().to_string();
    let output_template = temp_dir.join(format!("{token_prefix}.%(ext)s"));
    let output_template = output_template.to_string_lossy().into_owned();

    let output = Command::new("yt-dlp")
        .arg("--ignore-config")
        .arg("--no-playlist")
        .arg("--playlist-items")
        .arg("1")
        .arg("--max-downloads")
        .arg("1")
        .arg("--no-part")
        .arg("--no-progress")
        .arg("--format")
        .arg(YTDLP_FORMAT_SELECTOR)
        .arg("--max-filesize")
        .arg(MAX_UPLOAD_SIZE.to_string())
        .arg("-o")
        .arg(output_template)
        .arg("--")
        .arg(url.as_str())
        .output()
        .await;

    let output = match output {
        Ok(output) => output,
        Err(err) => {
            cleanup_prefixed_files(&temp_dir, &token_prefix);
            let message = if err.kind() == std::io::ErrorKind::NotFound {
                "yt-dlp executable not found".to_string()
            } else {
                err.to_string()
            };
            return Err(ApiError::DownloaderFailure(SmallString::new(message)));
        }
    };

    let files = list_prefixed_files(&temp_dir, &token_prefix)?;
    if files.is_empty() && !output.status.success() {
        cleanup_prefixed_files(&temp_dir, &token_prefix);
        let message = extract_downloader_error(&output);
        return Err(ApiError::DownloaderFailure(SmallString::new(message)));
    }
    if files.is_empty() {
        return Err(ApiError::DownloaderFailure(SmallString::new(
            "Downloader did not produce any files.",
        )));
    }
    let selected = match select_downloaded_file(&files) {
        Ok(selected) => selected,
        Err(err) => {
            cleanup_prefixed_files(&temp_dir, &token_prefix);
            return Err(err);
        }
    };
    if !output.status.success() {
        let message = extract_downloader_error(&output);
        eprintln!("yt-dlp exited with error but produced a file: {message}");
    }
    cleanup_unused_files(&files, &selected);

    let extension = selected
        .extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or("");
    MimeType::from_extension(extension)?;

    let token = selected.file_name().and_then(|name| name.to_str()).ok_or_else(|| {
        ApiError::DownloaderFailure(SmallString::new(
            "Downloader returned invalid filename",
        ))
    })?;

    Ok(token.to_string())
}

fn normalize_content_type(content_type: &str) -> String {
    let content_type = content_type
        .split(';')
        .next()
        .unwrap_or("")
        .trim()
        .to_ascii_lowercase();
    if content_type == "image/jpg" {
        "image/jpeg".to_string()
    } else {
        content_type
    }
}

fn should_infer_mime_from_url(content_type: &str) -> bool {
    content_type.is_empty()
        || content_type == "application/octet-stream"
        || content_type == "binary/octet-stream"
}

fn mime_type_from_url(url: &Url) -> Option<MimeType> {
    if let Some(segments) = url.path_segments() {
        if let Some(name) = segments.last() {
            if let Some((_, extension)) = name.rsplit_once('.') {
                if let Ok(mime_type) = MimeType::from_extension(extension) {
                    return Some(mime_type);
                }
            }
        }
    }

    url.query_pairs()
        .find(|(key, _)| key == "format")
        .and_then(|(_, value)| MimeType::from_extension(value.as_ref()).ok())
}

fn list_prefixed_files(temp_dir: &Path, prefix: &str) -> std::io::Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    for entry in std::fs::read_dir(temp_dir)? {
        let entry = entry?;
        let path = entry.path();
        let file_name = path.file_name().and_then(|name| name.to_str()).unwrap_or("");
        if file_name.starts_with(prefix) {
            files.push(path);
        }
    }
    Ok(files)
}

fn select_downloaded_file(files: &[PathBuf]) -> ApiResult<PathBuf> {
    let mut best: Option<(PathBuf, u64)> = None;
    for path in files {
        if MimeType::from_path(path).is_none() {
            continue;
        }
        let size = std::fs::metadata(path)?.len();
        let replace = match best {
            Some((_, best_size)) => size > best_size,
            None => true,
        };
        if replace {
            best = Some((path.clone(), size));
        }
    }

    best.map(|(path, _)| path).ok_or_else(|| {
        ApiError::DownloaderFailure(SmallString::new(
            "Downloader did not produce a supported file.",
        ))
    })
}

fn cleanup_prefixed_files(temp_dir: &Path, prefix: &str) {
    if let Ok(files) = list_prefixed_files(temp_dir, prefix) {
        for path in files {
            let _ = std::fs::remove_file(path);
        }
    }
}

fn cleanup_unused_files(files: &[PathBuf], keep: &Path) {
    for path in files {
        if path != keep {
            let _ = std::fs::remove_file(path);
        }
    }
}

fn extract_downloader_error(output: &std::process::Output) -> String {
    let stderr = String::from_utf8_lossy(&output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr_lines: Vec<&str> = stderr
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect();
    let stdout_lines: Vec<&str> = stdout
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect();

    let message = pick_downloader_message(&stderr_lines)
        .or_else(|| pick_downloader_message(&stdout_lines))
        .or_else(|| tail_downloader_message(&stderr_lines))
        .or_else(|| tail_downloader_message(&stdout_lines))
        .unwrap_or("yt-dlp failed to download content");

    message.to_string()
}

fn pick_downloader_message<'a>(lines: &'a [&'a str]) -> Option<&'a str> {
    lines
        .iter()
        .rev()
        .find(|line| line.contains("ERROR:"))
        .copied()
        .or_else(|| {
            lines
                .iter()
                .rev()
                .find(|line| line.contains("WARNING:"))
                .copied()
        })
        .or_else(|| lines.last().copied())
}

fn tail_downloader_message(lines: &[&str]) -> Option<String> {
    if lines.is_empty() {
        return None;
    }
    let start = lines.len().saturating_sub(5);
    Some(lines[start..].join(" | "))
}
