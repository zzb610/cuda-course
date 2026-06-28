#!/usr/bin/env node
import { createRequire } from "node:module";
import { execFileSync } from "node:child_process";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { basename, dirname, join, resolve } from "node:path";

const require = createRequire(import.meta.url);
const { Resvg } = require(resolveNpxDependency("@resvg/resvg-js"));

const root = resolve(dirname(new URL(import.meta.url).pathname), "..");
const fontCandidates = [
  join(root, "assets/fonts/Excalifont-Regular.ttf"),
  join(root, "assets/fonts/Excalifont-Regular.woff2"),
  "/Users/bowenyuchi/Documents/speculative-decoding-study/assets/fonts/Excalifont-Regular.woff2",
  "/Users/bowenyuchi/.vscode/extensions/dynow.excalimath-editor-1.1.1/webview/dist/fonts/Excalifont/Excalifont-Regular-be310b9bcd4f1a43f571c46df7809174.woff2",
];

const fontFile = fontCandidates.find((path) => existsSync(path));
if (!fontFile) {
  console.error("Cannot find Excalifont-Regular.ttf or Excalifont-Regular.woff2.");
  process.exit(1);
}
const cjkFontFiles = [
  join(root, "assets/fonts/LXGWWenKai-Regular.ttf"),
].filter((path) => existsSync(path));

const inputArgs = process.argv.slice(2);
const inputs = inputArgs.length > 0
  ? inputArgs
  : ["diagrams/*.excalidraw"];

const files = expandInputs(inputs);
if (files.length === 0) {
  console.error("No .excalidraw files found.");
  process.exit(1);
}

const outputDir = join(root, "diagrams/exported");
mkdirSync(outputDir, { recursive: true });

for (const input of files) {
  const inputPath = resolve(root, input);
  const base = basename(inputPath, ".excalidraw");
  const outPath = join(outputDir, `${base}.png`);
  const svg = renderSvgWithExistingExporter(inputPath);
  const patchedSvg = forceExcalifont(svg);
  const png = new Resvg(patchedSvg, {
    fitTo: { mode: "zoom", value: 2 },
    font: {
      fontFiles: [fontFile, ...cjkFontFiles],
      loadSystemFonts: true,
    },
  }).render().asPng();
  writeFileSync(outPath, png);
  console.log(`✓ ${outPath}`);
}

function expandInputs(patterns) {
  const result = [];
  for (const pattern of patterns) {
    if (pattern.includes("*")) {
      const dir = pattern.slice(0, pattern.indexOf("*"));
      const suffix = pattern.slice(pattern.indexOf("*") + 1);
      const absDir = resolve(root, dir);
      const names = execFileSync("find", [absDir, "-maxdepth", "1", "-type", "f", "-name", `*${suffix}`], {
        encoding: "utf8",
      }).trim();
      if (names) {
        result.push(...names.split("\n").map((path) => path.replace(`${root}/`, "")));
      }
    } else {
      result.push(pattern);
    }
  }
  return [...new Set(result)].sort();
}

function renderSvgWithExistingExporter(inputPath) {
  const tempDir = mkdtempSync(join(tmpdir(), "cuda-course-excalidraw-"));
  const svgPath = join(tempDir, "diagram.svg");
  try {
    execFileSync("npx", ["--yes", "@moona3k/excalidraw-export", inputPath, "-o", svgPath, "--svg"], {
      cwd: root,
      stdio: ["ignore", "pipe", "pipe"],
    });
    return readFileSync(svgPath, "utf8");
  } finally {
    rmSync(tempDir, { recursive: true, force: true });
  }
}

function forceExcalifont(svg) {
  const family = "Excalifont, LXGW WenKai, HanziPen SC, Kaiti SC, STKaiti, Segoe UI Emoji, cursive";
  return svg
    .replace(/font-family="Virgil, Segoe UI Emoji, cursive"/g, `font-family="${family}"`)
    .replace(/font-family="Virgil"/g, `font-family="${family}"`);
}

function resolveNpxDependency(packageName) {
  try {
    return require.resolve(packageName);
  } catch {
    const home = process.env.HOME;
    if (!home) {
      throw new Error(`Cannot resolve ${packageName}: HOME is not set.`);
    }
    const packagePath = join("node_modules", ...packageName.split("/"), "index.js");
    const matches = execFileSync(
      "find",
      [join(home, ".npm/_npx"), "-path", `*/${packagePath}`, "-type", "f"],
      { encoding: "utf8" },
    ).trim();
    const first = matches.split("\n").filter(Boolean)[0];
    if (!first) {
      throw new Error(`Cannot resolve ${packageName}. Run: npx --yes @moona3k/excalidraw-export --help`);
    }
    return first;
  }
}
