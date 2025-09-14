const socket = io();
const video = document.getElementById("video");
const localStatus = document.getElementById("localStatus");
const emotionResult = document.getElementById("emotionResult");
const chatbox = document.getElementById("chatbox");
const messageInput = document.getElementById("messageInput");
const sendBtn = document.getElementById("sendBtn");

// Register user
let username = prompt("Enter your name:") || "Anonymous";
socket.emit("register", { username });

// Access webcam
navigator.mediaDevices.getUserMedia({ video: true, audio: false })
  .then(stream => {
    video.srcObject = stream;

    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d");

    setInterval(() => {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      const data = canvas.toDataURL("image/jpeg", 0.5);
      socket.emit("frame", data);
    }, 1000); // 1 FPS
  })
  .catch(err => console.error("Camera error:", err));

// Handle prediction
socket.on("prediction", data => {
  if (!data) return;

  const label = data.label || "Unknown";
  const conf = typeof data.confidence === "number" ? data.confidence.toFixed(1) : "";

  if (data.category === "Active") {
    localStatus.textContent = "Active";
    localStatus.style.backgroundColor = "green";
  } else {
    localStatus.textContent = "Inactive";
    localStatus.style.backgroundColor = "red";
  }

  emotionResult.textContent = `Detected: ${label} ${conf ? "(" + conf + "%)" : ""}`;
});

// Chat
sendBtn.addEventListener("click", sendMessage);
messageInput.addEventListener("keypress", e => {
  if (e.key === "Enter") sendMessage();
});

function sendMessage() {
  const msg = messageInput.value.trim();
  if (!msg) return;
  socket.emit("chat", { username, message: msg });
  messageInput.value = "";
}

socket.on("chat", data => {
  const div = document.createElement("div");
  div.textContent = `${data.username}: ${data.message}`;
  chatbox.appendChild(div);
  chatbox.scrollTop = chatbox.scrollHeight;
});
