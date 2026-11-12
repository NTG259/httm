// Chụp web demo (http://localhost:8000) theo kịch bản 09_DEMO_SCRIPT §2 → docs/assets/demo_*.png
// node shots.js <outDir> [--no-llm]
// Cần puppeteer: cài kèm mermaid-cli (PUPPETEER_SKIP_DOWNLOAD=1 npm i @mermaid-js/mermaid-cli) rồi chạy với NODE_PATH
// trỏ tới node_modules đó. Dùng Chrome có sẵn trên máy. Bật API (uvicorn :8000) trước khi chạy.
const puppeteer = require("puppeteer");
const out = process.argv[2];
const useLlm = !process.argv.includes("--no-llm");
const CHROME = "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe";

const SCENES = {
  story1: ["Dạo này tôi khát nước suốt, đi tiểu nhiều, sụt 3 kg trong tháng.", "Tôi 52 tuổi.",
           "Tôi cao 1m65, nặng 78 kg.", "Tôi là nam.", "HbA1c tháng trước 7,1%."],
  tuvan: ["Tiêu chuẩn chẩn đoán tiểu đường là gì?"],
  capcuu: ["Tôi đường huyết 25 mmol/L, đang buồn nôn, thở dốc"],
};

(async () => {
  const browser = await puppeteer.launch({ executablePath: CHROME, headless: "new",
                                           defaultViewport: { width: 1500, height: 1000, deviceScaleFactor: 1 } });
  const page = await browser.newPage();
  await page.goto("http://localhost:8000", { waitUntil: "networkidle0" });
  await page.waitForFunction(() => !document.querySelector("#health").textContent.includes("đang kiểm tra"),
                             { timeout: 60000 });
  const box = await page.$("#useLlm");
  if ((await (await box.getProperty("checked")).jsonValue()) !== useLlm) await box.click();
  for (const [name, msgs] of Object.entries(SCENES)) {
    await page.click("#newBtn");
    await new Promise(r => setTimeout(r, 800));
    for (const m of msgs) {
      const before = await page.$$eval("#chat > *", els => els.length);
      await page.type("#text", m);
      await page.keyboard.press("Enter");
      // chờ tin của người dùng + phản hồi hệ thống (2 phần tử mới)
      await page.waitForFunction(n => document.querySelectorAll("#chat > *").length >= n + 2, { timeout: 240000 }, before);
      await new Promise(r => setTimeout(r, 500));
      console.log(`[${name}] ${m}`);
    }
    await page.evaluate(() => document.querySelectorAll("details").forEach(d => d.open = true));
    const file = `${out}/demo_${name}.png`;
    await page.screenshot({ path: file, fullPage: true });
    console.log("→", file);
  }
  await browser.close();
})().catch(e => { console.error(e); process.exit(1); });
