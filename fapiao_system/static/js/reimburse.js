/* 发票报销页脚本：筛选查看、导出选中、标记已报销/恢复未报销/标记已处理。 */
"use strict";

let records = [];
const checkedIds = new Set();
const pager = { page: 1, pageSize: 20 };

const statusFilter = document.getElementById("status-filter");
statusFilter.addEventListener("change", () => {
  pager.page = 1;
  loadRecords();
});

async function loadRecords() {
  const status = statusFilter.value;
  const data = await api(`api/admin/records${status ? `?status=${status}` : ""}`);
  records = data.records;
  const rows = paginate(records, pager);
  document.getElementById("records-empty").classList.toggle("hidden", records.length > 0);
  const tbody = document.querySelector("#records-table tbody");
  tbody.innerHTML = rows.map((r) => `
    <tr>
      <td><input type="checkbox" class="row-check" data-id="${r.id}" ${checkedIds.has(r.id) ? "checked" : ""}></td>
      <td>${esc(r.student_id)}</td>
      <td>${esc(r.user_name)}</td>
      <td>${esc(r.product_name)}</td>
      <td>${CHANNEL_TEXT[r.channel]}</td>
      <td class="num">${fmtMoney(r.amount)}</td>
      <td>${esc(r.paid_at)}</td>
      <td>${PAYER_TEXT[r.payer]}</td>
      <td><span class="badge ${STATUS_CLASS[r.status]}">${STATUS_TEXT[r.status]}</span></td>
      <td class="num">${r.invoices.length}</td>
      <td><button class="btn btn-sm" data-detail="${r.id}">查看详情</button></td>
    </tr>`).join("");

  const checkAll = document.getElementById("check-all");
  const pageIds = rows.map((r) => r.id);
  checkAll.checked = pageIds.length > 0 && pageIds.every((id) => checkedIds.has(id));
  checkAll.indeterminate = !checkAll.checked && pageIds.some((id) => checkedIds.has(id));
  refreshPagination();
}

const refreshPagination = () =>
  renderPagination(document.getElementById("records-pagination"), pager, records.length, loadRecords, checkedIds.size);

document.querySelector("#records-table tbody").addEventListener("click", (e) => {
  const btn = e.target.closest("button[data-detail]");
  if (!btn) return;
  openRecordDetail(Number(btn.dataset.detail)).catch((err) => toast(err.message, false));
});

const selectedIds = () => [...checkedIds];

const statusOf = (id) => records.find((r) => r.id === id)?.status;

document.getElementById("check-all").addEventListener("change", (e) => {
  paginate(records, pager).forEach((r) => {
    if (e.target.checked) checkedIds.add(r.id);
    else checkedIds.delete(r.id);
  });
  document.querySelectorAll(".row-check").forEach((cb) => {
    cb.checked = e.target.checked;
  });
  refreshPagination();
});

document.querySelector("#records-table tbody").addEventListener("change", (e) => {
  const cb = e.target.closest("input.row-check");
  if (!cb) return;
  if (cb.checked) checkedIds.add(Number(cb.dataset.id));
  else checkedIds.delete(Number(cb.dataset.id));
  refreshPagination();
});

document.getElementById("btn-reimburse").addEventListener("click", () => {
  const ids = selectedIds().filter((id) => statusOf(id) === "invoiced");
  if (!ids.length) {
    toast("选中的记录中没有已开票的记录", false);
    return;
  }
  openNoteActionModal({
    title: `标记已报销（${ids.length} 条）`,
    label: "报销备注（选填，200 字以内）",
    placeholder: "如：已随 9 月第二批材料提交财务",
    buttonText: "确认标记",
    endpoint: "api/admin/reimburse",
    successMsg: "已标记 {n} 条记录为已报销",
    ids,
  });
});

function openNoteActionModal({ title, label, placeholder, emptyHint, buttonText, endpoint, successMsg, ids }) {
  const overlay = showModal(title, `
    <div class="field">
      <span class="field-label">${esc(label)}</span>
      <textarea name="note" rows="3" maxlength="200"
        placeholder="${esc(placeholder)}"></textarea>
    </div>
    <p class="form-error" id="note-error"></p>
    <button type="button" class="btn btn-primary">${esc(buttonText)}</button>`);
  overlay.querySelector(".btn-primary").addEventListener("click", async () => {
    const note = overlay.querySelector("textarea").value.trim();
    const errEl = overlay.querySelector("#note-error");
    if (emptyHint && !note) {
      errEl.textContent = emptyHint;
      return;
    }
    try {
      const { updated } = await api(endpoint, { json: { ids, note } });
      closeModal();
      toast(successMsg.replace("{n}", updated));
      checkedIds.clear();
      loadRecords();
    } catch (err) {
      errEl.textContent = err.message;
    }
  });
}

document.getElementById("btn-reject").addEventListener("click", () => {
  const ids = selectedIds().filter((id) => statusOf(id) === "invoiced");
  if (!ids.length) {
    toast("选中的记录中没有已开票的记录", false);
    return;
  }
  openNoteActionModal({
    title: `驳回记录（${ids.length} 条）`,
    label: "驳回理由（必填，200 字以内，用户可见）",
    placeholder: "如：发票抬头与单位名称不符，请重新开票后上传",
    emptyHint: "请填写驳回理由",
    buttonText: "确认驳回",
    endpoint: "api/admin/reject",
    successMsg: "已驳回 {n} 条记录，等待用户修改",
    ids,
  });
});

document.getElementById("btn-revert").addEventListener("click", async () => {
  const ids = selectedIds().filter((id) => ["reimbursed", "rejected", "processed"].includes(statusOf(id)));
  if (!ids.length) {
    toast("请勾选已报销、已驳回或已处理的记录", false);
    return;
  }
  if (!confirm(`确定将选中的 ${ids.length} 条记录恢复为未报销吗？`)) return;
  try {
    const { updated } = await api("api/admin/revert", { json: { ids } });
    toast(`已恢复 ${updated} 条记录`);
    checkedIds.clear();
    loadRecords();
  } catch (err) {
    toast(err.message, false);
  }
});

document.getElementById("btn-process").addEventListener("click", () => {
  const ids = selectedIds().filter((id) => ["pending", "invoiced"].includes(statusOf(id)));
  if (!ids.length) {
    toast("请勾选待开票或已开票的记录", false);
    return;
  }
  openNoteActionModal({
    title: `标记已处理（${ids.length} 条）`,
    label: "处理说明（必填，200 字以内）",
    placeholder: "如：经手人与商家线下协商退货退款，无法走正常报销流程",
    emptyHint: "请填写处理说明",
    buttonText: "确认标记",
    endpoint: "api/admin/process",
    successMsg: "已将 {n} 条记录标记为已处理",
    ids,
  });
});

document.getElementById("btn-export").addEventListener("click", async () => {
  const ids = selectedIds();
  if (!ids.length) {
    toast("请先勾选要导出的记录", false);
    return;
  }
  try {
    await downloadFile(`api/admin/export?ids=${ids.join(",")}`, "报销材料.zip");
  } catch (err) {
    toast(err.message, false);
  }
});

loadMe("admin").then((user) => {
  if (user) loadRecords();
});
