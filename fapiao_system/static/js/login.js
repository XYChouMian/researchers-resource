/* 登录页脚本。 */
"use strict";

document.getElementById("login-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const fd = new FormData(e.target);
  const errEl = document.getElementById("login-error");
  errEl.textContent = "";
  try {
    await api("api/login", {
      json: { student_id: fd.get("student_id").trim(), password: fd.get("password") },
    });
    location.href = "user";
  } catch (err) {
    errEl.textContent = err.message;
  }
});

fetch("api/me")
  .then((res) => {
    if (res.ok) {
      return res.json().then(() => {
        location.href = "user";
      });
    }
  })
  .catch(() => {});
