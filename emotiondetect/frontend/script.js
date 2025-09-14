// Change IP if connecting from another device on LAN
const socket = io("http://127.0.0.1:5000");

const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const predictionText = document.getElementById("prediction-text");

// Chat elements
const chatInput = document.getElementById("chat-input");
const sendBtn = document.getElementById("send-btn");
const messagesDiv = document.getElementById("messages");

// Ask user for a name
let username = prompt("Enter your name:");
if (!username) username = "Anonymous_" + Math.floor(Math.random() * 1000);
socket.emit("register", { username });

// Handle predictions
socket.on("prediction", (data) => {
    predictionText.innerText = `${data.label} (${data.confidence.toFixed(1)}%)`;
});

// Handle chat
socket.on("chat", (data) => {
    const msg = document.createElement("p");
    msg.textContent = `${data.username}: ${data.message}`;
    messagesDiv.appendChild(msg);
    messagesDiv.scrollTop = messagesDiv.scrollHeight; // auto scroll
});

// Send chat message
sendBtn.addEventListener("click", () => {
    const message = chatInput.value.trim();
    if (message !== "") {
        socket.emit("chat", { message });
        chatInput.value = "";
    }
});

// Video streaming
navigator.mediaDevices.getUserMedia({ video: true }).then((stream) => {
    video.srcObject = stream;

    setInterval(() => {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

        const dataUrl = canvas.toDataURL("image/jpeg");
        socket.emit("frame", dataUrl);
    }, 1000); // send 1 frame per second
});
