document.addEventListener('DOMContentLoaded', () => {
    const input = document.getElementById('user-input');
    const sendBtn = document.getElementById('send-btn');
    const outputArea = document.getElementById('output-area');
    const inputContainer = document.getElementById('input-container');
    const statusText = document.getElementById('status-text');
    const statusBadge = document.getElementById('status-badge');
    const tokenInput = document.getElementById('auth-token');

    // Restore token if saved or from URL
    const urlParams = new URLSearchParams(window.location.search);
    const urlToken = urlParams.get('token');

    if (urlToken) {
        tokenInput.value = urlToken;
        localStorage.setItem('hapa_llada_token', urlToken);
        // Clean URL
        window.history.replaceState({}, document.title, "/");
    } else {
        const savedToken = localStorage.getItem('hapa_llada_token');
        if (savedToken) {
            tokenInput.value = savedToken;
        }
    }

    // Save token on change
    tokenInput.addEventListener('input', () => {
        localStorage.setItem('hapa_llada_token', tokenInput.value.trim());
        checkHealth(); // Re-check health with new token
    });

    // Auto-resize textarea
    input.addEventListener('input', function () {
        this.style.height = 'auto';
        this.style.height = (this.scrollHeight) + 'px';
        if (this.value === '') this.style.height = 'auto';
    });

    // Send on Enter (but Shift+Enter for newline)
    input.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    sendBtn.addEventListener('click', sendMessage);

    async function sendMessage() {
        const text = input.value.trim();
        const token = tokenInput.value.trim();

        if (!text) return;
        if (!token) {
            alert("Please enter the Node Token from the server logs.");
            tokenInput.focus();
            return;
        }

        // UI Updates
        addMessage(text, 'user');
        input.value = '';
        input.style.height = 'auto';
        setLoading(true);

        // Apply LLaDA Chat Template
        const systemPrompt = "You are a helpful assistant. Please answer in English.";
        const formattedPrompt = `<role>SYSTEM</role>${systemPrompt}<|role_end|><role>HUMAN</role>${text}<|role_end|><role>ASSISTANT</role>`;

        try {
            const startTime = Date.now();

            // Note: Using new v1/completions endpoint for strict auth checking
            const response = await fetch('/v1/completions', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${token}`
                },
                body: JSON.stringify({
                    prompt: formattedPrompt,
                    max_tokens: 200,
                    temperature: 0.6
                })
            });

            if (!response.ok) {
                if (response.status === 401) {
                    throw new Error("Unauthorized. Check your Node Token.");
                }
                const err = await response.json().catch(() => ({ detail: response.statusText }));
                throw new Error(err.detail || 'Generation failed');
            }

            const data = await response.json();
            const duration = ((Date.now() - startTime) / 1000).toFixed(2);

            // Supports OAI style response
            const responseText = data.choices[0].text;
            addMessage(responseText, 'ai', duration);
        } catch (error) {
            addMessage(`**Error:** ${error.message}`, 'ai');
            statusText.textContent = 'Error';
            statusBadge.style.color = 'var(--error-color)';
            statusBadge.querySelector('.status-dot').style.backgroundColor = 'var(--error-color)';
        } finally {
            setLoading(false);
        }
    }

    function addMessage(text, role, duration = null) {
        const msgDiv = document.createElement('div');
        msgDiv.className = `message ${role}`;

        let content = text;
        if (role === 'ai') {
            content = marked.parse(text);
        } else {
            content = text.replace(/</g, "&lt;").replace(/>/g, "&gt;");
        }

        let metaHtml = '';
        if (duration) {
            metaHtml = `<div class="generation-meta">
                <span><i class="fa-solid fa-clock"></i> ${duration}s</span>
                <span><i class="fa-solid fa-bolt"></i> MLX Diff</span>
            </div>`;
        }

        msgDiv.innerHTML = (role === 'ai' ? `<div class="markdown-body">${content}</div>` : content) + metaHtml;

        outputArea.appendChild(msgDiv);
        outputArea.scrollTop = outputArea.scrollHeight;
    }

    function setLoading(isLoading) {
        if (isLoading) {
            inputContainer.classList.add('loading');
            input.disabled = true;
            sendBtn.disabled = true;
            statusText.textContent = 'Diffusing...';
        } else {
            inputContainer.classList.remove('loading');
            input.disabled = false;
            sendBtn.disabled = false;
            statusText.textContent = 'Online';
            input.focus();
        }
    }

    // Health Check with Auth
    async function checkHealth() {
        // /health is public in strict sense or not? 
        // User implementation of /health was: 
        // @app.get("/health") -> async def health() -> ... (No Depends)
        // correct, but /capabilities requires auth. Let's use /health for basic online, 
        // and /capabilities to verify auth.

        try {
            const res = await fetch('/health');
            if (res.ok) {
                const data = await res.json();
                if (data.model_loaded) {
                    statusText.textContent = 'Online';
                    statusBadge.style.color = 'var(--success-color)';
                    statusBadge.querySelector('.status-dot').style.backgroundColor = 'var(--success-color)';
                } else {
                    statusText.textContent = 'Loading Model...';
                    statusBadge.style.color = 'orange';
                    statusBadge.querySelector('.status-dot').style.backgroundColor = 'orange';
                }
            }
        } catch (err) {
            statusText.textContent = 'Offline';
            statusBadge.style.borderColor = 'var(--error-color)';
            statusBadge.querySelector('.status-dot').style.backgroundColor = 'gray';
        }
    }

    // Initial Check
    checkHealth();
    setInterval(checkHealth, 30000);
});
