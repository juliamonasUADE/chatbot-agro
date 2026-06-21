const API_URL = "https://chatbot-agro-production-c45e.up.railway.app";

function preguntar(texto) {
    document.getElementById("inputMsg").value = texto;
    enviar();
}

async function enviar() {
    const input  = document.getElementById("inputMsg");
    const chat   = document.getElementById("chat");
    const btn    = document.getElementById("btnEnviar");

    const texto  = input.value.trim();
    if (!texto) return;

    agregarBurbuja(texto, "usuario");
    input.value = "";
    btn.disabled = true;

    const spinnerWrap = document.createElement("div");
    spinnerWrap.className = "spinner-wrap";
    spinnerWrap.innerHTML = '<div class="spinner"></div>';
    chat.appendChild(spinnerWrap);
    chat.scrollTop = chat.scrollHeight;

    try {
        const res = await fetch(`${API_URL}/chat`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ texto })
        });
        const data = await res.json();
        spinnerWrap.remove();

        const modos = { dataset: "Dataset", rag: "RAG", sin_resultado: "Sin resultado" };

        let html = data.respuesta.replace(/\n/g, "<br>");
        if (data.fuentes && data.fuentes.length > 0 && data.fuentes[0] !== "dataset") {
            const fuentesTexto = data.fuentes.map(f => {
                if (f.includes("pdf")) return "Mapa Productivo 2022";
                if (f.includes("docx")) return "Info Producción Madariaga";
                return f;
            }).join(", ");
            html += `<div class="fuentes">Fuente: ${fuentesTexto}</div>`;
        }

        agregarBurbuja(html, "bot", true);

    } catch (err) {
        spinnerWrap.remove();
        agregarBurbuja("Error al conectar con el servidor. Verificá que el backend esté corriendo.", "bot");
    }

    btn.disabled = false;
    input.focus();
}

function agregarBurbuja(texto, tipo, esHTML = false) {
    const chat = document.getElementById("chat");
    const div  = document.createElement("div");
    div.className = `burbuja ${tipo}`;
    if (esHTML) div.innerHTML = texto;
    else        div.textContent = texto;
    chat.appendChild(div);
    chat.scrollTop = chat.scrollHeight;
}

document.getElementById("inputMsg").addEventListener("keypress", e => {
    if (e.key === "Enter") { e.preventDefault(); enviar(); }
});