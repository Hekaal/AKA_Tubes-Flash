(function () {
  const root = document.documentElement;
  const overlay = document.getElementById("overlay");
  const runForm = document.getElementById("runForm");
  const runBtn = document.getElementById("runBtn");
  const toggleTheme = document.getElementById("toggleTheme");
  const clearInputs = document.getElementById("clearInputs");

  // ===== Theme =====
  function setTheme(t) {
    root.setAttribute("data-theme", t);
    localStorage.setItem("theme", t);
  }

  const savedTheme = localStorage.getItem("theme");
  if (savedTheme === "light" || savedTheme === "dark") setTheme(savedTheme);
  else setTheme("dark");

  if (toggleTheme) {
    toggleTheme.addEventListener("click", () => {
      const curr = root.getAttribute("data-theme") || "dark";
      setTheme(curr === "dark" ? "light" : "dark");
    });
  }

  // ===== Clear input =====
  if (clearInputs) {
    clearInputs.addEventListener("click", () => {
      const nList = document.getElementById("nList");
      const prefixes = document.getElementById("prefixes");
      if (nList) nList.value = "";
      if (prefixes) prefixes.value = "";
      nList?.focus();
    });
  }

  // ===== Validate + Loading overlay =====
  function validNList(s) {
    // allow digits, commas, spaces only
    return /^[0-9,\s]+$/.test(s.trim()) && s.trim().length > 0;
  }

  function validPrefixes(s) {
    return s.trim().length > 0;
  }

  if (runForm) {
    runForm.addEventListener("submit", (e) => {
      const nList = document.getElementById("nList")?.value || "";
      const prefixes = document.getElementById("prefixes")?.value || "";

      // Simple validation
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

      // Show overlay
      overlay?.classList.add("show");
      if (runBtn) {
        runBtn.disabled = true;
        runBtn.textContent = "Running...";
      }
    });
  }

  // Hide overlay if page restored from bfcache
  window.addEventListener("pageshow", () => {
    overlay?.classList.remove("show");
    if (runBtn) {
      runBtn.disabled = false;
      runBtn.textContent = "RUN Eksperimen";
    }
  });
})();
