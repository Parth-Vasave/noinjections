// After the page has loaded, fetch a note from a relative URL and add it to the page.
window.addEventListener("load", () => {
  setTimeout(() => {
    fetch("rendered-note.txt")
      .then((response) => response.text())
      .then((text) => {
        const note = document.createElement("p");
        note.className = "injected";
        note.textContent = text.trim();
        document.getElementById("notes").append(note);
      });
  }, 50);
});
