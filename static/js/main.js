document.addEventListener("DOMContentLoaded", () => {
  const drop = document.getElementById("drop");
  const file = document.getElementById("file");
  const preview = document.getElementById("preview");
  const chooseBtn = document.getElementById("chooseBtn");

  if (!drop || !file) return;

  const showPreview = (f) => {
    if (!f) return;
    const reader = new FileReader();
    reader.onload = (e) => {
      preview.src = e.target.result;
      preview.classList.remove("hidden");
      preview.classList.add("fade-in");
    };
    reader.readAsDataURL(f);
  };

  drop.addEventListener("click", () => file.click());
  chooseBtn?.addEventListener("click", () => file.click());

  drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("bg-white/30"); });
  drop.addEventListener("dragleave", () => drop.classList.remove("bg-white/30"));
  drop.addEventListener("drop", (e) => {
    e.preventDefault(); drop.classList.remove("bg-white/30");
    const f = e.dataTransfer.files[0];
    if (f) { file.files = e.dataTransfer.files; showPreview(f); }
  });

  file.addEventListener("change", (e) => showPreview(e.target.files[0]));
});
