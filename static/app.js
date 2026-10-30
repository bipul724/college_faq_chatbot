const chat = document.getElementById("chat"), q = document.getElementById("q");
function add(cls, text) {
  const d = document.createElement("div"); d.className = "msg " + cls; d.textContent = text;
  chat.appendChild(d); d.scrollIntoView({behavior: "smooth", block: "end"}); return d;
}
async function ask(text) {
  add("user", text);
  const wait = add("bot", "Searching documents…");
  try {
    const r = await fetch("/api/chat", {method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({question: text})});
    const data = await r.json();
    wait.textContent = data.answer || data.error || "Something went wrong.";
    if (data.sources && data.sources.length) {
      const det = document.createElement("details");
      const s = document.createElement("summary"); s.textContent = "Sources (" + data.sources.length + ")";
      det.appendChild(s);
      data.sources.forEach(x => { const p = document.createElement("p");
        p.textContent = x.source + (x.page != null ? " p." + (x.page + 1) : "") + " – match " + x.score;
        det.appendChild(p); });
      wait.appendChild(det);
    }
  } catch (e) { wait.textContent = "Could not reach the server. Is Flask running?"; }
}
document.getElementById("f").addEventListener("submit", e => {
  e.preventDefault(); const t = q.value.trim(); if (t) { q.value = ""; ask(t); }
});
document.querySelectorAll(".chip").forEach(c => c.addEventListener("click", () => ask(c.textContent)));
