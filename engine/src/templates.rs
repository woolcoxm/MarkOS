//! Chat-template rendering. Per-model custom Jinja2-style templates
//! (minijinja) take priority, then the GGUF's embedded `tokenizer.chat_template`,
//! then a generic fallback. The engine feeds the backend a plain prompt.

use minijinja::{context, Environment};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

pub struct TemplateInput<'a> {
    pub messages: &'a [ChatMessage],
    pub add_generation_prompt: bool,
}

/// Render the final prompt for the backend.
pub fn render_chat(
    custom_template: Option<&str>,
    gguf_template: Option<&str>,
    input: &TemplateInput,
) -> Result<String, String> {
    if let Some(tpl) = custom_template {
        return render_jinja(tpl, input)
            .map_err(|e| format!("custom chat template error: {e}"));
    }
    if let Some(tpl) = gguf_template {
        return render_jinja(tpl, input).map_err(|e| format!("gguf chat template error: {e}"));
    }
    Ok(render_fallback(input))
}

fn render_jinja(tpl: &str, input: &TemplateInput) -> Result<String, String> {
    let env = Environment::new();
    // HF templates occasionally use odd whitespace control; minijinja's
    // defaults handle the mainstream ones (ChatML, Llama-3, Gemma, Mistral).
    let messages: Vec<serde_json::Value> = input
        .messages
        .iter()
        .map(|m| serde_json::json!({"role": m.role, "content": m.content}))
        .collect();
    env.render_str(
        tpl,
        context! { messages => messages, add_generation_prompt => input.add_generation_prompt },
    )
    .map_err(|e| e.to_string())
}

fn render_fallback(input: &TemplateInput) -> String {
    let mut out = String::new();
    for m in input.messages {
        match m.role.as_str() {
            "system" => out.push_str(&format!("System: {}\n", m.content)),
            "user" => out.push_str(&format!("User: {}\n", m.content)),
            "assistant" => out.push_str(&format!("Assistant: {}\n", m.content)),
            other => out.push_str(&format!("{other}: {}\n", m.content)),
        }
    }
    if input.add_generation_prompt {
        out.push_str("Assistant:");
    }
    out
}

/// Render a template for preview in the web UI (no backend round-trip).
#[allow(dead_code)] // exposed for the web UI preview endpoint
pub fn preview(
    custom_template: Option<&str>,
    gguf_template: Option<&str>,
    messages: &[ChatMessage],
) -> Result<String, String> {
    render_chat(custom_template, gguf_template, &TemplateInput { messages, add_generation_prompt: true })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn msgs() -> Vec<ChatMessage> {
        vec![
            ChatMessage { role: "system".into(), content: "You are terse.".into() },
            ChatMessage { role: "user".into(), content: "Hi".into() },
        ]
    }

    #[test]
    fn chatml_style_template() {
        let tpl = "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}";
        let out = render_chat(Some(tpl), None, &TemplateInput { messages: &msgs(), add_generation_prompt: true }).unwrap();
        assert!(out.starts_with("<|im_start|>system\nYou are terse.<|im_end|>\n"));
        assert!(out.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn gguf_template_used_when_no_override() {
        let out = render_chat(None, Some("U: {{ messages[0].content }}"), &TemplateInput { messages: &msgs(), add_generation_prompt: false }).unwrap();
        assert_eq!(out, "U: You are terse.");
    }

    #[test]
    fn fallback_when_none() {
        let out = render_chat(None, None, &TemplateInput { messages: &msgs(), add_generation_prompt: true }).unwrap();
        assert!(out.contains("System: You are terse."));
        assert!(out.ends_with("Assistant:"));
    }

    #[test]
    fn broken_template_is_an_error_not_a_panic() {
        assert!(render_chat(Some("{% for %}"), None, &TemplateInput { messages: &msgs(), add_generation_prompt: true }).is_err());
    }
}
