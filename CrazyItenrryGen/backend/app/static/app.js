const form = document.getElementById("plan-form");
const errorEl = document.getElementById("error");
const resultSection = document.getElementById("result");
const titleEl = document.getElementById("result-title");
const notesEl = document.getElementById("result-notes");
const itemsEl = document.getElementById("items");

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  errorEl.textContent = "";
  resultSection.classList.add("hidden");
  itemsEl.innerHTML = "";

  const payload = {
    user_id: "local_demo",
    preference: {
      city: document.getElementById("city").value,
      days: Number(document.getElementById("days").value),
      chaos_level: Number(document.getElementById("chaos").value),
      interests: document.getElementById("interests").value.split(",").map((item) => item.trim()),
    },
  };

  try {
    const response = await fetch("/api/itineraries", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      throw new Error("Unable to generate itinerary.");
    }

    const data = await response.json();
    titleEl.textContent = `Itinerary for ${data.destination} (${data.total_days} days)`;
    notesEl.textContent = data.notes;

    data.itinerary.forEach((item) => {
      const card = document.createElement("div");
      card.className = "item-card";
      card.innerHTML = `
        <h3>${item.title}</h3>
        <p><strong>Day ${item.day}</strong> • ${item.category}</p>
        <p>${item.description}</p>
        <p><em>Location: ${item.location}</em></p>
      `;
      itemsEl.appendChild(card);
    });

    resultSection.classList.remove("hidden");
  } catch (err) {
    errorEl.textContent = err.message;
  }
});
