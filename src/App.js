import { useEffect, useState, useRef } from "react";

export default function App() {
  const [data, setData] = useState(null);
  const canvasRef = useRef(null);
  const imgRef = useRef(null);

  useEffect(() => {
    const interval = setInterval(() => {
      fetch("http://localhost:5000/latest")
        .then(res => res.json())
        .then(setData)
        .catch(console.error);
    }, 500);

    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    if (!data || !data.image) return;

    const img = new Image();
    img.src = `data:image/jpeg;base64,${data.image}`;
    img.onload = () => {
      const canvas = canvasRef.current;
      const ctx = canvas.getContext("2d");

      canvas.width = img.width;
      canvas.height = img.height;

      ctx.drawImage(img, 0, 0);

      // Draw boxes
      data.objects.forEach(obj => {
        const [x1, y1, x2, y2] = obj.box;

        ctx.strokeStyle = "lime";
        ctx.lineWidth = 2;
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

        ctx.fillStyle = "lime";
        ctx.font = "14px Arial";
        ctx.fillText(
          `${obj.label} (${obj.confidence})`,
          x1,
          y1 - 5
        );
      });
    };
  }, [data]);

  return (
    <div style={{ display: "flex", padding: "20px", fontFamily: "Arial" }}>
      <div>
        <h2>📷 Live CCTV</h2>
        <canvas ref={canvasRef} />
      </div>

      <div style={{ marginLeft: "20px", width: "300px" }}>
        <h2>🧠 AI Descriptions</h2>

        {data?.objects?.length === 0 && <p>No objects detected</p>}

        {data?.objects?.map((obj, i) => (
          <div key={i} style={{
            marginBottom: "10px",
            padding: "10px",
            background: "#111",
            color: "#0f0",
            borderRadius: "8px"
          }}>
            <strong>{obj.label}</strong> ({obj.confidence})
            <p>{obj.description}</p>
          </div>
        ))}

        <h4>Last Update:</h4>
        <p>
          {data?.timestamp
            ? new Date(data.timestamp * 1000).toLocaleTimeString()
            : "N/A"}
        </p>
      </div>
    </div>
  );
}