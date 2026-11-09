const messageBox = document.getElementById("message");

function showMessage(text) {
    messageBox.textContent = text;
    messageBox.classList.remove("hidden");
}

function clearMessage() {
    messageBox.textContent = "";
    messageBox.classList.add("hidden");
}

function percent(value) {
    return `${(Number(value) * 100).toFixed(1)}%`;
}

document.getElementById("individual-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    clearMessage();
    const resultsContainer = document.getElementById("individual-results");
    resultsContainer.innerHTML = "<p class='muted'>Calculando...</p>";
    try {
        const response = await fetch("/api/predict", { method: "POST", body: new FormData(event.target) });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || "No fue posible calcular la predicción.");
        resultsContainer.innerHTML = Object.entries(data.predictions).map(([model, result]) => {
            const high = result.class === 1;
            return `<article class="result ${high ? "high" : "low"}">
                <h3>${model}</h3>
                <div class="${high ? "high-text" : "low-text"}"><strong>${high ? "Mayor riesgo" : "Menor riesgo"}</strong></div>
                <div class="probability">${percent(result.probability)}</div>
                <div class="muted">${result.class_label}</div>
            </article>`;
        }).join("") + `<div class="score"><strong>Puntaje complementario:</strong> ${data.score}/7 — ${data.score_label}</div>`;
    } catch (error) {
        resultsContainer.innerHTML = "";
        showMessage(error.message);
    }
});

function metricCards(metrics) {
    return Object.entries(metrics).map(([key, value]) => `<span><strong>${key.toUpperCase()}</strong>: ${Number(value).toFixed(3)}</span>`).join(" · ");
}

function confusionTable(matrix) {
    return `<div class="confusion"><strong>Matriz de confusión</strong><table>
        <tr><th></th><th>Pred. no complicado</th><th>Pred. complicado</th></tr>
        <tr><th>Real no complicado</th><td>${matrix[0][0]}</td><td>${matrix[0][1]}</td></tr>
        <tr><th>Real complicado</th><td>${matrix[1][0]}</td><td>${matrix[1][1]}</td></tr>
    </table></div>`;
}

document.getElementById("batch-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    clearMessage();
    const container = document.getElementById("batch-results");
    container.innerHTML = "<p class='muted batch-heading'>Procesando archivo...</p>";
    try {
        const response = await fetch("/api/predict-batch", { method: "POST", body: new FormData(event.target) });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || "No fue posible procesar el archivo.");
        let html = `<h3 class="batch-heading">${data.rows} registros procesados</h3>`;
        for (const [model, result] of Object.entries(data.models)) {
            html += `<div class="metric-panel"><h3>${model}</h3>`;
            if (result.evaluation) {
                html += `<p class="muted">${metricCards(result.evaluation.metrics)}</p>${confusionTable(result.evaluation.confusion_matrix)}`;
            }
            html += `</div>`;
        }
        if (!data.has_target) html += `<p class="muted">El archivo no contiene la variable real; se muestran solo predicciones.</p>`;
        if (data.preview.length) {
            const columns = Object.keys(data.preview[0]);
            html += `<div class="table-wrap"><table><tr>${columns.map(column => `<th>${column}</th>`).join("")}</tr>`;
            html += data.preview.slice(0, 20).map(row => `<tr>${columns.map(column => `<td>${row[column] ?? ""}</td>`).join("")}</tr>`).join("");
            html += `</table></div><p class="muted">Se muestran máximo 20 filas en pantalla.</p>`;
        }
        container.innerHTML = html;
    } catch (error) {
        container.innerHTML = "";
        showMessage(error.message);
    }
});
