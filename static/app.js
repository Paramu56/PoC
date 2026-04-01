function el(id) {
  return document.getElementById(id);
}

async function ask() {
  const payload = {
    question: el("question").value,
    profile: {
      name: el("name").value,
      gender: el("gender").value,
      age: el("age").value,
      caste: el("caste").value,
      residence: el("residence").value,
      marital_status: el("marital_status").value,
      disability_percentage: el("disability_percentage").value,
      employment_status: el("employment_status").value,
      occupation: el("occupation").value,
      minority: el("minority").value,
      below_poverty_line: el("below_poverty_line").value,
      economic_distress: el("economic_distress").value
    }
  };

  el("status").innerText = "Running retrieval + Gemini orchestration...";
  el("answer").style.display = "none";
  el("citations").style.display = "none";
  el("debug").style.display = "none";

  try {
    const r = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    const data = await r.json();
    if (!r.ok) throw new Error(data.error || "Request failed");

    el("answer").style.display = "block";
    el("answer").innerHTML =
      "<h3>Answer</h3><div>" + (data.answer || "").replaceAll("\n", "<br/>") + "</div>";

    el("citations").style.display = "block";
    el("citations").innerHTML =
      "<h3>Citations</h3>" +
      (data.citations || [])
        .map((c) => `<div>- ${c.scheme_name || "Unknown"} (page ${c.page || "?"})</div>`)
        .join("");

    el("debug").style.display = "block";
    el("debug").innerHTML = "<h3>Debug</h3><pre>" + JSON.stringify(data.debug || {}, null, 2) + "</pre>";

    el("status").innerText = "Done.";
  } catch (e) {
    el("status").innerText = "Error: " + e.message;
  }
}

document.addEventListener("DOMContentLoaded", () => {
  const btn = el("askBtn");
  btn.addEventListener("click", ask);

  el("question").addEventListener("keydown", (ev) => {
    if (ev.key === "Enter" && (ev.ctrlKey || ev.metaKey)) {
      ask();
    }
  });
});

