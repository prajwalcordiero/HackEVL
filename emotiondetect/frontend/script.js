const video = document.getElementById("localVideo");
const statusDiv = document.getElementById("status");

// Start video
navigator.mediaDevices.getUserMedia({ video: true, audio: true })
  .then((stream) => {
    video.srcObject = stream;

    // Capture a frame every 2s
    setInterval(() => {
      sendFrameToServer(video);
    }, 2000);
  });

async function sendFrameToServer(video) {
  // Draw video frame to canvas
  const canvas = document.createElement("canvas");
  canvas.width = 48;
  canvas.height = 48;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(video, 0, 0, 48, 48);

  // Convert to grayscale
  let imgData = ctx.getImageData(0, 0, 48, 48);
  for (let i = 0; i < imgData.data.length; i += 4) {
    let avg = (imgData.data[i] + imgData.data[i+1] + imgData.data[i+2]) / 3;
    imgData.data[i] = imgData.data[i+1] = imgData.data[i+2] = avg;
  }
  ctx.putImageData(imgData, 0, 0);

  // Encode to base64
  const imageBase64 = canvas.toDataURL("image/jpeg").split(",")[1];

  // Send to Flask
  const res = await fetch("http://localhost:5001/infer", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ image: imageBase64 })
  });

  const result = await res.json();
  statusDiv.innerText = `Detected: ${result.label} (${result.confidence}%)`;
}
