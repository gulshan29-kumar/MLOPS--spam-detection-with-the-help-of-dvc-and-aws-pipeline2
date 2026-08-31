document.addEventListener('DOMContentLoaded', () => {
    const messageInput = document.getElementById('message-input');
    const charsCount = document.getElementById('chars');
    const analyzeBtn = document.getElementById('analyze-btn');
    const clearBtn = document.getElementById('clear-btn');
    const btnSpinner = document.getElementById('btn-spinner');
    
    const resultsPanel = document.getElementById('results-panel');
    const predictionBadge = document.getElementById('prediction-badge');
    const outcomeDesc = document.getElementById('outcome-desc');
    const tokensList = document.getElementById('preprocessed-tokens');

    // Real-time character counter
    messageInput.addEventListener('input', () => {
        const count = messageInput.value.length;
        charsCount.textContent = count;
    });

    // Clear input interface
    clearBtn.addEventListener('click', () => {
        messageInput.value = '';
        charsCount.textContent = '0';
        resultsPanel.style.display = 'none';
        messageInput.focus();
    });

    // Run prediction query
    analyzeBtn.addEventListener('click', async () => {
        const textValue = messageInput.value.trim();
        if (!textValue) {
            alert('Please enter some text to analyze.');
            return;
        }

        // Toggle state to Loading
        analyzeBtn.disabled = true;
        clearBtn.disabled = true;
        messageInput.disabled = true;
        btnSpinner.style.display = 'block';
        analyzeBtn.querySelector('span').textContent = 'Analyzing...';
        
        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ text: textValue })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            
            if (data.success) {
                // Populate results
                renderResults(data);
            } else {
                alert('Prediction failed. Please try again.');
            }
        } catch (error) {
            console.error('Error during prediction:', error);
            alert('Failed to connect to the prediction API. Check console logs for details.');
        } finally {
            // Restore button visual states
            btnSpinner.style.display = 'none';
            analyzeBtn.disabled = false;
            clearBtn.disabled = false;
            messageInput.disabled = false;
            analyzeBtn.querySelector('span').textContent = 'Analyze Message';
        }
    });

    function renderResults(data) {
        // Reset classes
        predictionBadge.classList.remove('spam', 'ham');
        
        const label = data.prediction.toUpperCase();
        predictionBadge.textContent = label;
        
        if (label === 'SPAM') {
            predictionBadge.classList.add('spam');
            outcomeDesc.textContent = 'This message has a high probability of containing spam content, promotional ads, or phishing triggers.';
        } else {
            predictionBadge.classList.add('ham');
            outcomeDesc.textContent = 'This message matches safe user characteristics (ham) and is likely safe to read.';
        }

        // Clear previous token chips
        tokensList.innerHTML = '';
        
        if (data.cleaned_text) {
            const tokens = data.cleaned_text.split(' ').filter(t => t.length > 0);
            if (tokens.length > 0) {
                tokens.forEach(token => {
                    const tag = document.createElement('span');
                    tag.className = 'token-tag';
                    tag.textContent = token;
                    tokensList.appendChild(tag);
                });
            } else {
                tokensList.innerHTML = '<span class="detail-val" style="font-size:0.9rem; color:var(--text-muted);">No tokens extracted. Only special symbols/numbers were parsed.</span>';
            }
        } else {
            tokensList.innerHTML = '<span class="detail-val" style="font-size:0.9rem; color:var(--text-muted);">No tokens reported by preprocessor.</span>';
        }

        // Show the panel
        resultsPanel.style.display = 'block';
        
        // Scroll to results panel
        resultsPanel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
});
