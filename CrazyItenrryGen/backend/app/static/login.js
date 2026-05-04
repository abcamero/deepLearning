const loginForm = document.getElementById("login-form");

loginForm.addEventListener("submit", (event) => {
  event.preventDefault();
  const email = document.getElementById("email").value.trim();
  const password = document.getElementById("password").value.trim();

  if (!email || !password) {
    alert("Please enter both email and password.");
    return;
  }

  window.localStorage.setItem("crazy_itinerary_user", JSON.stringify({ email }));
  window.location.href = "/";
});
