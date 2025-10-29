// Navigation
document.addEventListener("DOMContentLoaded", function () {
  const navItems = document.querySelectorAll(".nav-item");
  const sections = document.querySelectorAll(".content-section");
  const menuToggle = document.getElementById("menuToggle");
  const sidebar = document.querySelector(".sidebar");

  loadBibFilesDropdown();

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

// Muestra/oculta y carga sólo al abrir el dropdown:
document
  .getElementById("userDropdownButton")
  .addEventListener("click", function (e) {
    e.preventDefault();
    const dropdown = this.closest(".user-dropdown");
    dropdown.classList.toggle("open");
    const menu = dropdown.querySelector(".user-dropdown-menu");
    if (dropdown.classList.contains("open")) {
      menu.innerHTML = "<li>Cargando...</li>";
      loadBibFilesDropdown();
    } else {
      menu.innerHTML = "";
    }
  });

// Oculta el dropdown si haces click afuera
document.addEventListener("click", function (e) {
  const dropdown = document.querySelector(".user-dropdown");
  if (dropdown && !dropdown.contains(e.target)) {
    dropdown.classList.remove("open");
  }
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
      const lines = String(data.result).split('\n').filter(line => line.trim());
      if (lines.length > 0) {
        let tableHtml = '<table class="table table-striped" style="margin:1em 0;min-width:300px;">';
        tableHtml += '<thead><tr><th>Métrica</th><th>Valor</th></tr></thead><tbody>';
        lines.forEach(line => {
          const [key, value] = line.split(':').map(s => s.trim());
          if (key && value !== undefined) {
            tableHtml += `<tr><td>${key}</td><td>${value}</td></tr>`;
          }
        });
        tableHtml += '</tbody></table>';
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

      // Display PDF links
      const pdfDiv = document.getElementById("bibliometric-pdf");
      pdfDiv.innerHTML = "";

      if (data.pdfs && data.pdfs.length > 0) {
        data.pdfs.forEach((pdfUrl) => {
          const filename = pdfUrl.split("/").pop();
          const link = document.createElement("a");
          link.href = pdfUrl;
          link.className = "pdf-link";
          link.download = filename;
          link.innerHTML = `<i class="fas fa-file-pdf"></i> ${filename}`;
          pdfDiv.appendChild(link);
        });
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

async function loadBibFilesDropdown() {
  try {
    const response = await fetch("/list_bib_files");
    const data = await response.json();
    console.log("Archivos .bib:", data); // Debug
    const list = document.getElementById("bib-file-list");
    list.innerHTML = "";
    for (const [source, files] of Object.entries(data.files)) {
      if (files.length > 0) {
        const groupLi = document.createElement("li");
        groupLi.innerHTML = `<strong>${source}:</strong>`;
        list.appendChild(groupLi);
        files.forEach((file) => {
          const li = document.createElement("li");
          li.style.marginLeft = "1.5em";
          li.innerHTML = `<a href="/static/data/raw/${source}/${file}" target="_blank">${file}</a>`;
          list.appendChild(li);
        });
      }
    }
  } catch (e) {
    showMessage(
      "merge-result",
      "Error al cargar archivos .bib: " + e.message,
      "error"
    );
  } finally {
    hideLoading();
  }
}

// async function runSentimentAnalysis() {
//   showLoading();
//   try {
//     const response = await fetch('/sentiment', { method: 'POST' });
//     const data = await response.json();
//     if (data.status === 'success') {
//       showMessage('sentiment-result', data.message, 'success');
//       document.getElementById('sentiment-output').innerHTML = /* genera el HTML dinámico */;
//     } else {
//       showMessage('sentiment-result', data.message, 'error');
//     }
//   } finally { hideLoading(); }
// }

// async function runCoAuthorshipAnalysis() {
//   showLoading();
//   try {
//     const response = await fetch('/coauthorship', { method: 'POST' });
//     const data = await response.json();
//     if (data.status === 'success') {
//       showMessage('coauthorship-result', data.message, 'success');
//       document.getElementById('coauthorship-output').innerHTML = /* HTML dinámico */;
//     } else {
//       showMessage('coauthorship-result', data.message, 'error');
//     }
//   } finally { hideLoading(); }
// }
