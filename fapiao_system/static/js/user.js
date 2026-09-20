/* 用户页脚本：提交购买记录、上传发票、编辑/删除待开票记录。 */
"use strict";

let me = null;
let records = [];

const form = document.getElementById("record-form");
const noteField = document.getElementById("channel-note-field");
const recordsFilter = document.getElementById("records-filter");
const checkedIds = new Set();
const pager = { page: 1, pageSize: 20 };
recordsFilter.addEventListener("change", () => {
  pager.page = 1;
  loadRecords();
});

document.getElementById("channel-select").addEventListener("change", (e) => {
  noteField.classList.toggle("hidden", e.target.value !== "other");
});

form.paid_at.value = todayLocal();

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  try {
    await api("api/records", { json: collectForm(form) });
    form.reset();
    form.paid_at.value = todayLocal();
    noteField.classList.add("hidden");
    toast("记录已提交");
    loadRecords();
  } catch (err) {
    toast(err.message, false);
  }
});

function collectForm(root) {
  return {
    product_name: root.product_name.value.trim(),
    channel: root.channel.value,
    channel_note: root.channel_note.value.trim(),
    remark: root.remark.value.trim(),
    amount: root.amount.value,
    paid_at: root.paid_at.value,
    payer: root.payer.value,
  };
}

async function loadRecords() {
  const data = await api("api/records");
  records = data.records;
  const shown = filteredRecords();
  document.getElementById("records-empty").classList.toggle("hidden", shown.length > 0);
  const rows = paginate(shown, pager);
  const tbody = document.querySelector("#records-table tbody");
  tbody.innerHTML = rows.map((r) => `
    <tr>
      <td><input type="checkbox" class="row-check" data-id="${r.id}" ${checkedIds.has(r.id) ? "checked" : ""}></td>
      <td>${esc(r.product_name)}</td>
      <td>${CHANNEL_TEXT[r.channel]}${r.channel_note ? `（${esc(r.channel_note)}）` : ""}</td>
      <td class="num">${fmtMoney(r.amount)}</td>
      <td>${esc(r.paid_at)}</td>
      <td>${PAYER_TEXT[r.payer]}</td>
      <td><span class="badge ${STATUS_CLASS[r.status]}">${STATUS_TEXT[r.status]}</span></td>
      <td class="actions">${renderActions(r)}</td>
      <td class="num">${r.invoices.length}</td>
      <td><button class="btn btn-sm" data-detail="${r.id}">查看详情</button></td>
    </tr>`).join("");

  const checkAll = document.getElementById("check-all");
  const pageIds = rows.map((r) => r.id);
  checkAll.checked = pageIds.length > 0 && pageIds.every((id) => checkedIds.has(id));
  checkAll.indeterminate = !checkAll.checked && pageIds.some((id) => checkedIds.has(id));
  refreshPagination();
}

function filteredRecords() {
  const allowed = recordsFilter.value ? recordsFilter.value.split(",") : null;
  return allowed ? records.filter((r) => allowed.includes(r.status)) : records;
}

const refreshPagination = () =>
  renderPagination(document.getElementById("records-pagination"), pager, filteredRecords().length, loadRecords, checkedIds.size);

const statusOf = (id) => records.find((r) => r.id === id)?.status;

document.getElementById("check-all").addEventListener("change", (e) => {
  paginate(filteredRecords(), pager).forEach((r) => {
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

const DELETABLE_STATUSES = ["pending", "invoiced", "rejected"];

document.getElementById("btn-batch-delete").addEventListener("click", async () => {
  const ids = [...checkedIds].filter((id) => DELETABLE_STATUSES.includes(statusOf(id)));
  if (!ids.length) {
    toast("选中的记录中没有可删除的记录（已报销/已处理不可删）", false);
    return;
  }
  if (!confirm(`确定删除选中的 ${ids.length} 条记录吗？删除后不可恢复`)) return;
  try {
    const { deleted } = await api("api/records/batch-delete", { json: { ids } });
    toast(`已删除 ${deleted} 条记录`);
    checkedIds.clear();
    loadRecords();
  } catch (err) {
    toast(err.message, false);
  }
});

document.getElementById("btn-export").addEventListener("click", async () => {
  const ids = [...checkedIds];
  if (!ids.length) {
    toast("请先勾选要导出的记录", false);
    return;
  }
  try {
    await downloadFile(`api/records/export?ids=${ids.join(",")}`, "我的报销材料.zip");
  } catch (err) {
    toast(err.message, false);
  }
});

function renderActions(r) {
  if (["reimbursed", "processed"].includes(r.status)) {
    return '<span class="muted">已锁定</span>';
  }
  const uploadLabel = r.status === "pending" ? "上传发票" : "补充附件";
  return [
    `<button class="btn btn-sm" data-act="upload" data-id="${r.id}">${uploadLabel}</button>`,
    `<button class="btn btn-sm" data-act="edit" data-id="${r.id}">编辑</button>`,
    `<button class="btn btn-sm btn-danger" data-act="delete" data-id="${r.id}">删除</button>`,
  ].join(" ");
}

document.querySelector("#records-table tbody").addEventListener("click", (e) => {
  const detailBtn = e.target.closest("button[data-detail]");
  if (detailBtn) {
    openRecordDetail(Number(detailBtn.dataset.detail)).catch((err) => toast(err.message, false));
    return;
  }
  const btn = e.target.closest("button[data-act]");
  if (!btn) return;
  const rec = records.find((r) => r.id === Number(btn.dataset.id));
  if (btn.dataset.act === "upload") openUploadModal(rec);
  if (btn.dataset.act === "edit") openEditModal(rec);
  if (btn.dataset.act === "delete") deleteRecord(rec);
});

async function deleteRecord(rec) {
  if (!confirm(`确定删除记录「${rec.product_name}」吗？`)) return;
  try {
    await api(`api/records/${rec.id}`, { method: "DELETE" });
    toast("已删除");
    loadRecords();
  } catch (err) {
    toast(err.message, false);
  }
}

function openUploadModal(rec) {
  const overlay = showModal(`发票与附件 - ${rec.product_name}`, `
    <div id="upload-count" class="upload-total"></div>
    ${rec.invoices.length ? `
      <table class="data-table upload-table">
        <thead><tr><th>已上传文件</th><th>类型</th><th>备注</th><th></th></tr></thead>
        <tbody>
          ${rec.invoices.map((inv) => `
            <tr>
              <td><a href="api/invoices/${inv.id}/download" title="${esc(inv.orig_name)}">${esc(shortName(inv.orig_name, 16))}</a></td>
              <td><span class="badge ${inv.category === "invoice" ? "st-invoiced" : "st-pending"}">${FILE_TYPE_TEXT[inv.category] ?? "附件"}</span></td>
              <td class="remark-cell">${esc(inv.remark || "")}</td>
              <td><button type="button" class="btn btn-sm btn-danger" data-del-inv="${inv.id}">删除</button></td>
            </tr>`).join("")}
        </tbody>
      </table>` : ''}
    <table class="data-table upload-table">
      <thead><tr><th>新文件（PDF/JPG/PNG，≤20MB）</th><th>类型</th><th>备注（选填）</th><th></th></tr></thead>
      <tbody id="new-rows"></tbody>
    </table>
    <div class="upload-actions">
      <button type="button" id="add-row" class="btn btn-sm">+ 添加附件</button>
      <button type="button" id="upload-submit" class="btn btn-primary">上传</button>
    </div>
    <p class="form-error" id="upload-error"></p>`);
  overlay.querySelector(".modal").classList.add("modal-wide");

  const newRows = overlay.querySelector("#new-rows");
  const countEl = overlay.querySelector("#upload-count");
  const errEl = overlay.querySelector("#upload-error");
  const existingInvoiceCount = () =>
    rec.invoices.filter((i) => i.category === "invoice").length;

  const refreshCount = () => {
    const newCount = [...newRows.querySelectorAll("select")]
      .filter((s) => s.value === "invoice").length;
    countEl.innerHTML =
      `开票张数：<b>${existingInvoiceCount() + newCount}</b>（自动计算 = 发票类文件数）`;
  };

  const addRow = () => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><input type="file" accept=".pdf,.jpg,.jpeg,.png" required></td>
      <td>
        <select>
          <option value="invoice">发票</option>
          <option value="attachment">附件</option>
        </select>
      </td>
      <td><input name="remark" maxlength="100" placeholder="选填"></td>
      <td><button type="button" class="btn btn-sm btn-danger">移除</button></td>`;
    tr.querySelector("button").addEventListener("click", () => {
      tr.remove();
      refreshCount();
    });
    tr.querySelector("select").addEventListener("change", refreshCount);
    newRows.appendChild(tr);
  };
  overlay.querySelector("#add-row").addEventListener("click", addRow);
  addRow();
  refreshCount();

  overlay.querySelectorAll("button[data-del-inv]").forEach((btn) => {
    btn.addEventListener("click", async () => {
      if (!confirm("确定删除这个文件吗？")) return;
      try {
        const { record } = await api(`api/invoices/${btn.dataset.delInv}`, { method: "DELETE" });
        rec.invoices = record.invoices;
        rec.status = record.status;
        rec.invoice_count = record.invoice_count;
        btn.closest("tr").remove();
        refreshCount();
        loadRecords();
      } catch (err) {
        errEl.textContent = err.message;
      }
    });
  });

  overlay.querySelector("#upload-submit").addEventListener("click", async () => {
    errEl.textContent = "";
    const rows = [...newRows.querySelectorAll("tr")];
    for (const tr of rows) {
      if (!tr.querySelector("input[type=file]").files.length) {
        errEl.textContent = "还有未选择文件的行，请选择文件或移除该行";
        return;
      }
    }
    const newInvoiceCount = rows.filter(
      (tr) => tr.querySelector("select").value === "invoice"
    ).length;
    if (existingInvoiceCount() + newInvoiceCount < 1) {
      errEl.textContent = "至少需要 1 个“发票”类型的文件";
      return;
    }
    if (!rows.length) {
      closeModal();
      return;
    }
    const fd = new FormData();
    for (const tr of rows) {
      fd.append("files", tr.querySelector("input[type=file]").files[0]);
      fd.append("categories", tr.querySelector("select").value);
      fd.append("remarks", tr.querySelector("input[name=remark]").value.trim());
    }
    try {
      await api(`api/records/${rec.id}/invoices`, { method: "POST", body: fd });
      closeModal();
      toast("上传成功");
      loadRecords();
    } catch (err) {
      errEl.textContent = err.message;
    }
  });
}

function openEditModal(rec) {
  const overlay = showModal(`编辑记录 - ${rec.product_name}`, `
    <form id="edit-form" class="form-col">
      <div class="field">
        <span class="field-label">商品名</span>
        <input name="product_name" required maxlength="200" value="${esc(rec.product_name)}">
      </div>
      <div class="field">
        <span class="field-label">购买渠道</span>
        <select name="channel">
          <option value="taobao">淘宝</option>
          <option value="jd">京东</option>
          <option value="other">其他</option>
        </select>
      </div>
      <div class="field" id="edit-note-field">
        <span class="field-label">渠道说明（必填）</span>
        <input name="channel_note" maxlength="100" value="${esc(rec.channel_note)}">
      </div>
      <div class="field">
        <span class="field-label">付款金额（元）</span>
        <input name="amount" type="number" min="0.01" step="0.01" required value="${rec.amount}">
      </div>
      <div class="field">
        <span class="field-label">付款时间</span>
        <input name="paid_at" type="date" required value="${esc(rec.paid_at)}">
      </div>
      <div class="field">
        <span class="field-label">付款人</span>
        <div class="radio-row">
          <label><input type="radio" name="payer" value="self"> 本人</label>
          <label><input type="radio" name="payer" value="tang"> 唐老师</label>
        </div>
      </div>
      <div class="field">
        <span class="field-label">备注（选填）</span>
        <input name="remark" maxlength="200" value="${esc(rec.remark || "")}">
      </div>
      <button type="submit" class="btn btn-primary">保存</button>
    </form>`);
  const editForm = overlay.querySelector("#edit-form");
  editForm.channel.value = rec.channel;
  editForm.payer.value = rec.payer;
  const editNoteField = overlay.querySelector("#edit-note-field");
  editNoteField.classList.toggle("hidden", rec.channel !== "other");
  editForm.channel.addEventListener("change", () => {
    editNoteField.classList.toggle("hidden", editForm.channel.value !== "other");
  });
  editForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    try {
      await api(`api/records/${rec.id}`, { method: "PUT", json: collectForm(editForm) });
      closeModal();
      toast("已保存");
      loadRecords();
    } catch (err) {
      toast(err.message, false);
    }
  });
}

loadMe().then((user) => {
  me = user;
  if (user) loadRecords();
});
