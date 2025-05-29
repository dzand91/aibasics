document.getElementById('upload-form').onsubmit = async function(e) {
    e.preventDefault();
    let fileInput = document.getElementById('pdf-file');
    if (fileInput.files.length === 0) return;
    let formData = new FormData();
    formData.append('file', fileInput.files[0]);
    await fetch('/upload', { method: 'POST', body: formData });
    document.getElementById('chat-window').innerHTML += `<div class="message bot">PDF uploaded! You can now ask questions about it.</div>`;
};

document.getElementById('chat-form').onsubmit = async function(e) {
    e.preventDefault();
    let userInput = document.getElementById('user-input');
    let msg = userInput.value.trim();
    if (!msg) return;
    let chatWindow = document.getElementById('chat-window');
    chatWindow.innerHTML += `<div class="message user">${msg}</div>`;
    userInput.value = '';
    let response = await fetch('/chat', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ message: msg })
    });
    let data = await response.json();
    chatWindow.innerHTML += `<div class="message bot">${data.answer}</div>`;
    chatWindow.scrollTop = chatWindow.scrollHeight;
};