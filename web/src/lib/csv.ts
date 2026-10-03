/**
 * Robust CSV parser and UNSW-NB15 flow batch generator for client-side ingestion.
 */

export interface ParsedFlow {
  flow_id: string;
  event_time: string;
  features: Record<string, string | number>;
  ground_truth?: {
    label?: number;
    attack_cat?: string;
  };
}

export interface ParseResult {
  flows: ParsedFlow[];
  errors: string[];
  totalRows: number;
  hasGroundTruth: boolean;
}

const INT_FIELDS = new Set([
  "spkts",
  "dpkts",
  "sbytes",
  "dbytes",
  "sttl",
  "dttl",
  "sloss",
  "dloss",
  "swin",
  "stcpb",
  "dtcpb",
  "dwin",
  "smean",
  "dmean",
  "trans_depth",
  "response_body_len",
  "ct_srv_src",
  "ct_state_ttl",
  "ct_dst_ltm",
  "ct_src_dport_ltm",
  "ct_dst_sport_ltm",
  "ct_dst_src_ltm",
  "is_ftp_login",
  "ct_ftp_cmd",
  "ct_flw_http_mthd",
  "ct_src_ltm",
  "ct_srv_dst",
  "is_sm_ips_ports",
]);

const FLOAT_FIELDS = new Set([
  "dur",
  "rate",
  "sload",
  "dload",
  "sinpkt",
  "dinpkt",
  "sjit",
  "djit",
  "tcprtt",
  "synack",
  "ackdat",
]);

const CATEGORY_FIELDS = new Set(["proto", "service", "state"]);

const ALL_REQUIRED_FEATURES = new Set([
  ...INT_FIELDS,
  ...FLOAT_FIELDS,
  ...CATEGORY_FIELDS,
]);

/**
 * Standard CSV line parser handling quotes and commas.
 */
function parseCSVLine(line: string): string[] {
  const result: string[] = [];
  let current = "";
  let inQuotes = false;

  for (let i = 0; i < line.length; i++) {
    const char = line[i];
    if (char === '"') {
      if (inQuotes && line[i + 1] === '"') {
        current += '"';
        i++;
      } else {
        inQuotes = !inQuotes;
      }
    } else if (char === "," && !inQuotes) {
      result.push(current.trim());
      current = "";
    } else {
      current += char;
    }
  }
  result.push(current.trim());
  return result;
}

/**
 * Parses raw CSV text into validated flows adhering to UNSW-NB15 schema.
 */
export function parseCSVText(csvText: string): ParseResult {
  const lines = csvText
    .split(/\r?\n/)
    .map((l) => l.trim())
    .filter((l) => l.length > 0);

  if (lines.length < 2) {
    return {
      flows: [],
      errors: ["CSV must contain a header row and at least one data row."],
      totalRows: 0,
      hasGroundTruth: false,
    };
  }

  const rawHeaders = parseCSVLine(lines[0]);
  const headers = rawHeaders.map((h) => h.toLowerCase().replace(/['"]/g, ""));

  // Check missing required columns
  const headerSet = new Set(headers);
  const missing = Array.from(ALL_REQUIRED_FEATURES).filter((f) => !headerSet.has(f));
  if (missing.length > 0) {
    return {
      flows: [],
      errors: [`Missing required feature columns: ${missing.slice(0, 8).join(", ")}${missing.length > 8 ? ` (+${missing.length - 8} more)` : ""}`],
      totalRows: lines.length - 1,
      hasGroundTruth: false,
    };
  }

  const hasLabel = headerSet.has("label");
  const hasCat = headerSet.has("attack_cat");
  const labelIdx = headers.indexOf("label");
  const catIdx = headers.indexOf("attack_cat");

  const flows: ParsedFlow[] = [];
  const errors: string[] = [];

  const nowIso = new Date().toISOString();

  for (let r = 1; r < lines.length; r++) {
    const values = parseCSVLine(lines[r]);
    if (values.length !== headers.length) {
      if (errors.length < 5) {
        errors.push(`Row ${r}: Column count mismatch (expected ${headers.length}, got ${values.length}).`);
      }
      continue;
    }

    const rowDict: Record<string, string> = {};
    for (let c = 0; c < headers.length; c++) {
      rowDict[headers[c]] = values[c];
    }

    const featDict: Record<string, string | number> = {};
    let rowValid = true;

    for (const feat of ALL_REQUIRED_FEATURES) {
      const rawVal = rowDict[feat];
      if (rawVal === undefined || rawVal === "") {
        if (errors.length < 5) {
          errors.push(`Row ${r}: Missing value for '${feat}'.`);
        }
        rowValid = false;
        break;
      }

      if (INT_FIELDS.has(feat)) {
        const parsed = parseInt(rawVal, 10);
        if (isNaN(parsed)) {
          if (errors.length < 5) errors.push(`Row ${r}: Invalid integer for '${feat}': ${rawVal}`);
          rowValid = false;
          break;
        }
        featDict[feat] = parsed;
      } else if (FLOAT_FIELDS.has(feat)) {
        const parsed = parseFloat(rawVal);
        if (isNaN(parsed)) {
          if (errors.length < 5) errors.push(`Row ${r}: Invalid number for '${feat}': ${rawVal}`);
          rowValid = false;
          break;
        }
        featDict[feat] = parsed;
      } else {
        // Category string
        featDict[feat] = rawVal.replace(/[^A-Za-z0-9_-]/g, "_") || "-";
      }
    }

    if (!rowValid) continue;

    const groundTruth: { label?: number; attack_cat?: string } = {};
    if (hasLabel && labelIdx !== -1) {
      const lbl = parseInt(values[labelIdx], 10);
      if (!isNaN(lbl)) groundTruth.label = lbl;
    }
    if (hasCat && catIdx !== -1) {
      groundTruth.attack_cat = values[catIdx];
    }

    flows.push({
      flow_id: crypto.randomUUID(),
      event_time: nowIso,
      features: featDict,
      ground_truth: groundTruth,
    });
  }

  return {
    flows,
    errors,
    totalRows: lines.length - 1,
    hasGroundTruth: hasLabel,
  };
}

/**
 * Splits array of parsed flows into batches of size <= maxBatchSize.
 */
export function chunkFlows<T>(items: T[], maxBatchSize = 100): T[][] {
  const chunks: T[][] = [];
  for (let i = 0; i < items.length; i += maxBatchSize) {
    chunks.push(items.slice(i, i + maxBatchSize));
  }
  return chunks;
}
