// Function to navigate between pages
function navigateTo(pageId) {
    // Hide all pages
    document.querySelectorAll('.page').forEach(page => {
        page.classList.remove('active');
    });
    
    // Show requested page
    document.getElementById(pageId).classList.add('active');
    
    // Toggle rope visibility based on page
    toggleRopeVisibility(pageId);
}

// Function to toggle rope visibility
function toggleRopeVisibility(pageId) {
    const ropeContainers = document.querySelectorAll('.rope-container');
    
    if (pageId === 'home') {
        ropeContainers.forEach(rope => {
            rope.style.display = 'block';
        });
    } else {
        ropeContainers.forEach(rope => {
            rope.style.display = 'none';
        });
    }
}

// Add click events to navigation links
document.querySelectorAll('.nav-link').forEach(link => {
    link.addEventListener('click', (e) => {
        e.preventDefault();
        const pageId = link.getAttribute('data-page');
        
        if (pageId === 'analysis') {
            // If clicking Analysis, show login first
            navigateTo('login');
        } else {
            navigateTo(pageId);
        }
    });
});

// Handle analyze button click
document.getElementById('analyze-btn').addEventListener('click', function() {
    const text = document.getElementById('text-input').value.trim();
    
    if (text) {
        // Send request to Flask backend
        fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
            },
            body: new URLSearchParams({
                'text': text
            })
        })
        .then(response => response.json())
        .then(data => {
            displayResult(data.sentiment, data.color, text);
        })
        .catch(error => {
            console.error('Error:', error);
        });
    }
});

// Function to display sentiment result
function displayResult(sentiment, color, text) {
    const resultsContainer = document.getElementById('results-container');
    const positiveResult = document.getElementById('positive-result');
    const negativeResult = document.getElementById('negative-result');
    const neutralResult = document.getElementById('neutral-result');
    const positiveText = document.getElementById('positive-text');
    const negativeText = document.getElementById('negative-text');
    const neutralText = document.getElementById('neutral-text');
    
    // Hide all results
    positiveResult.style.display = 'none';
    negativeResult.style.display = 'none';
    neutralResult.style.display = 'none';
    
    // Show appropriate result based on sentiment
    if (sentiment === 'Positive Sentiment') {
        positiveResult.style.display = 'block';
        positiveText.textContent = text;
    } else if (sentiment === 'Negative Sentiment') {
        negativeResult.style.display = 'block';
        negativeText.textContent = text;
    } else {
        neutralResult.style.display = 'block';
        neutralText.textContent = text;
    }
    
    // Show results container
    resultsContainer.style.display = 'block';
}

// Initialize page
window.addEventListener('DOMContentLoaded', function() {
    // Initial checks for rope visibility
    toggleRopeVisibility('home');
});