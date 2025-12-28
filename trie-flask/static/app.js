(function () {
  const root = document.documentElement;
  const overlay = document.getElementById("overlay");
  const runForm = document.getElementById("runForm");
  const runBtn = document.getElementById("runBtn");
  const toggleTheme = document.getElementById("toggleTheme");
  const clearInputs = document.getElementById("clearInputs");

  const useDefaultPath = document.getElementById("useDefaultPath");
  const csvPath = document.getElementById("csvPath");

  // ===== Theme =====
  function setTheme(t) {
    root.setAttribute("data-theme", t);
    localStorage.setItem("theme", t);
  }
  const savedTheme = localStorage.getItem("theme");
  if (savedTheme === "light" || savedTheme === "dark") setTheme(savedTheme);
  else setTheme("dark");

  toggleTheme?.addEventListener("click", () => {
    const curr = root.getAttribute("data-theme") || "dark";
    setTheme(curr === "dark" ? "light" : "dark");
  });

  // ===== Default path helper =====
  useDefaultPath?.addEventListener("click", () => {
    if (csvPath) csvPath.value = "amazon_products.csv";
    csvPath?.focus();
  });

  // ===== Clear input =====
  clearInputs?.addEventListener("click", () => {
    const nList = document.getElementById("nList");
    const prefixes = document.getElementById("prefixes");
    if (nList) nList.value = "";
    if (prefixes) prefixes.value = "";
    nList?.focus();
  });

  // ===== Validate + Loading overlay =====
  function validNList(s) {
    return /^[0-9,\s]+$/.test(s.trim()) && s.trim().length > 0;
  }
  function validPrefixes(s) {
    return s.trim().length > 0;
  }

  runForm?.addEventListener("submit", (e) => {
    const nList = document.getElementById("nList")?.value || "";
    const prefixes = document.getElementById("prefixes")?.value || "";

    if (!validNList(nList)) {
      e.preventDefault();
      alert("N list tidak valid. Contoh: 1000,5000,10000");
      return;
    }
    if (!validPrefixes(prefixes)) {
      e.preventDefault();
      alert("Prefix kosong. Isi minimal 1 baris prefix.");
      return;
    }

    overlay?.classList.add("show");
    if (runBtn) {
      runBtn.disabled = true;
      runBtn.textContent = "Running...";
    }
  });

  window.addEventListener("pageshow", () => {
    overlay?.classList.remove("show");
    if (runBtn) {
      runBtn.disabled = false;
      runBtn.textContent = "RUN Eksperimen";
    }
  });
})();
