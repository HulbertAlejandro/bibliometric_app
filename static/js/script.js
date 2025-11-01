// Navigation
document.addEventListener("DOMContentLoaded", function () {
  const navItems = document.querySelectorAll(".nav-item");
  const sections = document.querySelectorAll(".content-section");
  const menuToggle = document.getElementById("menuToggle");
  const sidebar = document.querySelector(".sidebar");

  renderBibFilesSection();
  
  // Navigation handling
  navItems.forEach((item) => {
    item.addEventListener("click", function (e) {
      e.preventDefault();

      // Remove active class from all items
      navItems.forEach((nav) => nav.classList.remove("active"));
      sections.forEach((section) => section.classList.remove("active"));

      // Add active class to clicked item
      this.classList.add("active");

      // Show corresponding section
      const sectionId = this.getAttribute("data-section");
      document.getElementById(sectionId).classList.add("active");

      // Close sidebar on mobile
      if (window.innerWidth <= 1024) {
        sidebar.classList.remove("active");
      }
    });
  });

  // Mobile menu toggle
  if (menuToggle) {
    menuToggle.addEventListener("click", function () {
      sidebar.classList.toggle("active");
    });
  }

  // Close sidebar when clicking outside on mobile
  document.addEventListener("click", function (e) {
    if (window.innerWidth <= 1024) {
      if (!sidebar.contains(e.target) && !menuToggle.contains(e.target)) {
        sidebar.classList.remove("active");
      }
    }
  });
});

// Utility Functions
function showLoading() {
  document.getElementById("loadingOverlay").classList.add("active");
}

function hideLoading() {
  document.getElementById("loadingOverlay").classList.remove("active");
}

function showMessage(elementId, message, type) {
  const element = document.getElementById(elementId);
  element.textContent = message;
  element.className = `result-message ${type}`;
  element.style.display = "block";

  setTimeout(() => {
    element.style.display = "none";
  }, 5000);
}

// ACM Scraper
async function runACMScraper() {
  const start = parseInt(document.getElementById("acm-start").value);
  const count = parseInt(document.getElementById("acm-count").value);

  showLoading();

  try {
    const response = await fetch("/scraper/acm", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ start, count }),
    });

    const data = await response.json();

    if (data.status === "success") {
      showMessage("acm-result", data.message, "success");
    } else {
      showMessage("acm-result", data.message, "error");
    }
  } catch (error) {
    showMessage(
      "acm-result",
      "Error al ejecutar el scraper: " + error.message,
      "error"
    );
  } finally {
    hideLoading();
  }
}

// IEEE Scraper
async function runIEEEScraper() {
  const start = parseInt(document.getElementById("ieee-start").value);
  const count = parseInt(document.getElementById("ieee-count").value);

  showLoading();

  try {
    const response = await fetch("/scraper/ieee", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ start, count }),
    });

    const data = await response.json();

    if (data.status === "success") {
      showMessage("ieee-result", data.message, "success");
    } else {
      showMessage("ieee-result", data.message, "error");
    }
  } catch (error) {
    showMessage(
      "ieee-result",
      "Error al ejecutar el scraper: " + error.message,
      "error"
    );
  } finally {
    hideLoading();
  }
}

// Merge BibTeX
async function runMerge() {
  showLoading();

  try {
    const response = await fetch("/merge", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
    });

    const data = await response.json();

    if (data.status === "success") {
      showMessage("merge-result", data.message, "success");
      // Mostrar resultados en el div correcto
      const filesDiv = document.getElementById("merge-files-links");
      filesDiv.innerHTML = "";

      if (data.csv_data) {
        for (const [filename, info] of Object.entries(data.csv_data)) {
          const section = document.createElement("div");
          let htmlSection = `<h4 style="margin: 1.5rem 0 1rem;">${filename}</h4>`;
          // Si hay tabla (tipo CSV), muestra tabla y botón de descarga
          if (info.table) {
            htmlSection += `${info.table}`;
          }
          section.innerHTML = htmlSection;
          filesDiv.appendChild(section);
        }
      }
    } else {
      showMessage("merge-result", data.message, "error");
    }
  } catch (error) {
    showMessage(
      "merge-result",
      "Error al fusionar archivos: " + error.message,
      "error"
    );
  } finally {
    hideLoading();
  }
}

// Compare Articles
async function runComparison() {
  const ids = document.getElementById("article-ids").value;
  const outputBox = document.getElementById("compare-output");

  // Oculta siempre al iniciar
  outputBox.style.display = "none";
  outputBox.innerHTML = "";

  if (!ids.trim()) {
    showMessage("compare-result", "Por favor ingrese al menos un ID", "error");
    return;
  }

  showLoading();

  try {
    const response = await fetch("/compare", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ ids }),
    });

    const data = await response.json();

    if (data.status === "success" && data.result) {
      showMessage("compare-result", data.message, "success");
      // Muestra solo si hay resultado, usando tabla formateada
      const lines = String(data.result)
        .split("\n")
        .filter((line) => line.trim());
      if (lines.length > 0) {
        let tableHtml =
          '<table class="table table-striped" style="margin:1em 0;min-width:300px;">';
        tableHtml +=
          "<thead><tr><th>Métrica</th><th>Valor</th></tr></thead><tbody>";
        lines.forEach((line) => {
          const [key, value] = line.split(":").map((s) => s.trim());
          if (key && value !== undefined) {
            tableHtml += `<tr><td>${key}</td><td>${value}</td></tr>`;
          }
        });
        tableHtml += "</tbody></table>";
        outputBox.innerHTML = tableHtml;
        outputBox.style.display = "block";
      }
    } else {
      showMessage("compare-result", data.message, "error");
      outputBox.style.display = "none";
    }
  } catch (error) {
    showMessage(
      "compare-result",
      "Error al comparar artículos: " + error.message,
      "error"
    );
    outputBox.style.display = "none";
  } finally {
    hideLoading();
  }
}

// Keywords Analysis
async function runKeywordsAnalysis() {
  showLoading();

  try {
    const response = await fetch("/keywords", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
    });

    const data = await response.json();

    if (data.status === "success") {
      showMessage("keywords-result", data.message, "success");

      const outputDiv = document.getElementById("keywords-output");
      outputDiv.innerHTML = "";

      if (data.images && data.images.length > 0) {
        data.images.forEach((imgUrl) => {
          const imgElement = document.createElement("div");
          imgElement.className = "gallery-item";
          imgElement.innerHTML = `<img src="${imgUrl}" alt="Keywords Analysis">`;
          outputDiv.appendChild(imgElement);
        });
      }
    } else {
      showMessage("keywords-result", data.message, "error");
    }
  } catch (error) {
    showMessage(
      "keywords-result",
      "Error al analizar keywords: " + error.message,
      "error"
    );
  } finally {
    hideLoading();
  }
}

// Dendrograms Analysis
async function runDendrogramsAnalysis() {
  showLoading();

  try {
    const response = await fetch("/dendrograms", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
    });

    const data = await response.json();

    if (data.status === "success") {
      showMessage("dendrograms-result", data.message, "success");

      // Display images
      const outputDiv = document.getElementById("dendrograms-output");
      outputDiv.innerHTML = "";

      if (data.images && data.images.length > 0) {
        data.images.forEach((imgUrl) => {
          const imgElement = document.createElement("div");
          imgElement.className = "gallery-item";
          imgElement.innerHTML = `<img src="${imgUrl}" alt="Dendrogram">`;
          outputDiv.appendChild(imgElement);
        });
      }

      // Display CSV data
      const csvDiv = document.getElementById("dendrograms-csv");
      csvDiv.innerHTML = "";

      if (data.csv_data) {
        for (const [filename, htmlTable] of Object.entries(data.csv_data)) {
          const section = document.createElement("div");
          section.innerHTML = `<h4 style="margin: 1.5rem 0 1rem;">${filename}</h4>${htmlTable}`;
          csvDiv.appendChild(section);
        }
      }
    } else {
      showMessage("dendrograms-result", data.message, "error");
    }
  } catch (error) {
    showMessage(
      "dendrograms-result",
      "Error al generar dendrogramas: " + error.message,
      "error"
    );
  } finally {
    hideLoading();
  }
}

// Bibliometric Analysis
async function runBibliometricAnalysis() {
  showLoading();

  try {
    const response = await fetch("/bibliometric", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
    });

    const data = await response.json();

    if (data.status === "success") {
      showMessage("bibliometric-result", data.message, "success");

      // Display images
      const outputDiv = document.getElementById("bibliometric-output");
      outputDiv.innerHTML = "";

      if (data.images && data.images.length > 0) {
        data.images.forEach((imgUrl) => {
          const imgElement = document.createElement("div");
          imgElement.className = "gallery-item";
          imgElement.innerHTML = `<img src="${imgUrl}" alt="Bibliometric Analysis">`;
          outputDiv.appendChild(imgElement);
        });
      }

      // Display PDF embedded
      const pdfDiv = document.getElementById("bibliometric-pdf");
      pdfDiv.innerHTML = "";

      if (data.pdfs && data.pdfs.length > 0) {
        // Usa solo el primer PDF si hay más de uno
        const pdfUrl = "/static/salidas/info_bibliometrica/" + data.pdfs[0];
        pdfDiv.innerHTML = `
          <div class="pdf-viewer">
            <h4>Reporte PDF Visualizable</h4>
            <iframe src="${pdfUrl}" width="1400px" height="700px" frameborder="0" style="border:1px solid #308113; margin-bottom:1rem;"></iframe>
            <a href="${pdfUrl}" class="pdf-link btn btn-secondary" download>
              <i class="fas fa-download"></i> Descargar PDF
            </a>
          </div>
        `;
      }

      // Display CSV data
      const csvDiv = document.getElementById("bibliometric-csv");
      csvDiv.innerHTML = "";

      if (data.csv_data) {
        for (const [filename, htmlTable] of Object.entries(data.csv_data)) {
          const section = document.createElement("div");
          section.innerHTML = `<h4 style="margin: 1.5rem 0 1rem;">${filename}</h4>${htmlTable}`;
          csvDiv.appendChild(section);
        }
      }
    } else {
      showMessage("bibliometric-result", data.message, "error");
    }
  } catch (error) {
    showMessage(
      "bibliometric-result",
      "Error al generar análisis: " + error.message,
      "error"
    );
  } finally {
    hideLoading();
  }
}

// Sentiment Analysis
async function runSentimentAnalysis() {
  showLoading();
  try {
    const response = await fetch("/sentiment", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
    });
    const data = await response.json();

    if (data.status === "success") {
      showMessage("sentiment-result", data.message, "success");
      const outputDiv = document.getElementById("sentiment-output");
      outputDiv.innerHTML = "";

      // Mostrar todas las imágenes como galería
      if (data.images && data.images.length > 0) {
        data.images.forEach((imgUrl) => {
          const imgElement = document.createElement("div");
          imgElement.className = "gallery-item";
          imgElement.innerHTML = `<img src="${imgUrl}" alt="Sentiment Analysis" />`;
          outputDiv.appendChild(imgElement);
        });
      }
    } else {
      showMessage("sentiment-result", data.message, "error");
    }
  } catch (error) {
    showMessage(
      "sentiment-result",
      "Error al ejecutar el análisis: " + error.message,
      "error"
    );
  } finally {
    hideLoading();
  }
}

// Co-authorship Analysis
async function runCoAuthorshipAnalysis() {
  showLoading();
  try {
    const response = await fetch("/grafos", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
    });
    const data = await response.json();

    if (data.status === "success") {
      showMessage("coauthorship-result", data.message, "success");

      // Mostrar todas las imágenes en galería
      const imagesDiv = document.getElementById("coauthorship-images");
      imagesDiv.innerHTML = "";
      if (data.images && data.images.length > 0) {
        data.images.forEach((imgUrl) => {
          const imgElement = document.createElement("div");
          imgElement.className = "gallery-item";
          imgElement.innerHTML = `<img src="${imgUrl}" alt="Red de Co-autoría">`;
          imagesDiv.appendChild(imgElement);
        });
      }

      // Mostrar los CSVs
      const csvDiv = document.getElementById("coauthorship-csv");
      csvDiv.innerHTML = "";
      if (data.csv_data) {
        for (const [filename, htmlTable] of Object.entries(data.csv_data)) {
          const section = document.createElement("div");
          section.innerHTML = `<h4 style="margin: 1.5rem 0 1rem;">${filename}</h4>${htmlTable}`;
          csvDiv.appendChild(section);
        }
      }
    } else {
      showMessage("coauthorship-result", data.message, "error");
    }
  } catch (error) {
    showMessage(
      "coauthorship-result",
      "Error al ejecutar el análisis: " + error.message,
      "error"
    );
  } finally {
    hideLoading();
  }
}

async function renderBibFilesSection() {
  const content = document.getElementById("bib-files-content");
  content.innerHTML =
    "<div style='color:#aaa;'>Cargando archivos .bib...</div>";
  try {
    const response = await fetch("/list_bib_files");
    const data = await response.json();

    let out = "";
    let total = 0;
    for (const [source, files] of Object.entries(data.files)) {
      if (files.length > 0) {
        out += `<h4 style="color:var(--secondary-color,#8b5cf6);margin:.5em 0;font-weight:700">${source}</h4><ul style="margin-bottom:1.2em;margin-left:1.5em;">`;
        files.forEach((file) => {
          out += `<li style="margin:.13em 0;">
            <a href="/static/data/raw/${source}/${file}" target="_blank" style="color:var(--primary-color,#6366f1);text-decoration:none;">${file}</a>
            <span style="color:#8e8e8e;font-size:0.91em;margin-left:.5em;">[descargar]</span>
          </li>`;
          total++;
        });
        out += "</ul>";
      }
    }
    if (!total)
      out =
        "<div style='color:#cdcdcd;'>No hay archivos .bib en ninguna fuente.</div>";
    content.innerHTML = out;
  } catch (e) {
    content.innerHTML = `<div style="color:#ef4444;">Error al cargar archivos .bib: ${
      e.message || e
    }</div>`;
  }
}

window.addEventListener("DOMContentLoaded", function () {
  var btnOpen = document.getElementById("openDeleteModal");
  var modal = document.getElementById("deleteFilesModal");
  var btnClose = document.getElementById("closeDeleteModal");
  var btnDeleteAll = document.getElementById("delete-all-files");
  var btnDeleteAnalysis = document.getElementById("delete-analysis-files");

  function showModal() {
    modal.classList.remove("fading-out");
    modal.classList.add("visible");
  }

  function hideModal() {
    modal.classList.add("fading-out");
    modal.classList.remove("visible");
    // Cuando termine la transición, quita el display por completo:
    setTimeout(function () {
      modal.classList.remove("fading-out");
      modal.style.display = "none";
    }, 340);
  }

  // Abrir modal
  if (btnOpen && modal) {
    btnOpen.onclick = function () {
      modal.style.display = "flex"; // Garantiza que está en el flow
      requestAnimationFrame(showModal); // Espera 1 frame y hace el fade
    };
  }

  // Cerrar modal por botón
  if (btnClose && modal) {
    btnClose.onclick = function () {
      hideModal();
    };
  }

  // Cerrar haciendo click fuera del cuadro
  if (modal) {
    modal.addEventListener("click", function (e) {
      if (e.target === modal) hideModal();
    });
  }

  // Acciones de borrado
  if (btnDeleteAll && modal) {
    btnDeleteAll.onclick = function () {
      borrarArchivos("all");
    };
  }
  if (btnDeleteAnalysis && modal) {
    btnDeleteAnalysis.onclick = function () {
      borrarArchivos("analysis");
    };
  }

  function borrarArchivos(tipo) {
    showLoading();
    fetch("/delete_files", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ type: tipo }),
    })
      .then((r) => r.json())
      .then((data) => {
        showSnackbar(data.message, 3300);
        hideModal();
        hideLoading();
      })
      .catch((e) => {
        showSnackbar("Error: " + e, 4000);
        hideLoading();
        hideModal();
      });
  }
});

function showSnackbar(msg, duration = 3200) {
  var sb = document.getElementById("snackbar");
  sb.innerText = msg;
  sb.classList.add("visible");
  setTimeout(function () {
    sb.classList.remove("visible");
    setTimeout(function () {
      sb.innerText = "";
      sb.style.display = "none";
    }, 400);
  }, duration);
  sb.style.display = "block";
}
