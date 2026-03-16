mkdir -p ~/.streamlit/

printf '%s\n' \
"[server]" \
"port = ${PORT}" \
"enableCORS = false" \
"headless = true" \
"" > ~/.streamlit/config.toml
