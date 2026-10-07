export const predictorFields = [
  "dur",
  "proto",
  "service",
  "state",
  "spkts",
  "dpkts",
  "sbytes",
  "dbytes",
  "rate",
  "sttl",
  "dttl",
  "sload",
  "dload",
  "sloss",
  "dloss",
  "sinpkt",
  "dinpkt",
  "sjit",
  "djit",
  "swin",
  "stcpb",
  "dtcpb",
  "dwin",
  "tcprtt",
  "synack",
  "ackdat",
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
] as const;

const categoricalFields = new Set(["proto", "service", "state"]);
const optionalLabels = ["label", "ground_truth", "attack_cat"];

function splitCsvLine(line: string) {
  const values: string[] = [];
  let current = "";
  let quoted = false;
  for (let index = 0; index < line.length; index += 1) {
    const character = line[index];
    if (character === '"') {
      if (quoted && line[index + 1] === '"') {
        current += '"';
        index += 1;
      } else {
        quoted = !quoted;
      }
    } else if (character === "," && !quoted) {
      values.push(current.trim());
      current = "";
    } else {
      current += character;
    }
  }
  values.push(current.trim());
  return values;
}

export interface ParsedCsv {
  rows: Record<string, string | number>[];
  labels: Array<"Attack" | "Normal" | undefined>;
  errors: string[];
}

export function parseNexusCsv(source: string): ParsedCsv {
  const lines = source.replace(/\r/g, "").split("\n").filter((line) => line.trim());
  if (lines.length < 2) return { rows: [], labels: [], errors: ["The CSV must include a header and at least one data row."] };

  const headers = splitCsvLine(lines[0]);
  const missing = predictorFields.filter((field) => !headers.includes(field));
  if (missing.length) {
    return {
      rows: [],
      labels: [],
      errors: missing.map((field) => `Missing required feature: ${field}`),
    };
  }

  const rows: Record<string, string | number>[] = [];
  const labels: Array<"Attack" | "Normal" | undefined> = [];
  const errors: string[] = [];
  const labelField = optionalLabels.find((field) => headers.includes(field));

  lines.slice(1).forEach((line, rowIndex) => {
    const values = splitCsvLine(line);
    if (values.length !== headers.length) {
      errors.push(`Row ${rowIndex + 2}: expected ${headers.length} columns, received ${values.length}.`);
      return;
    }
    const record: Record<string, string | number> = {};
    predictorFields.forEach((field) => {
      const raw = values[headers.indexOf(field)];
      if (raw === "") {
        errors.push(`Row ${rowIndex + 2}: missing value for ${field}.`);
      } else if (categoricalFields.has(field)) {
        record[field] = raw;
      } else {
        const numeric = Number(raw);
        if (!Number.isFinite(numeric)) errors.push(`Row ${rowIndex + 2}: ${field} must be numeric.`);
        else record[field] = numeric;
      }
    });
    if (Object.keys(record).length === predictorFields.length) {
      rows.push(record);
      if (!labelField) labels.push(undefined);
      else {
        const rawLabel = values[headers.indexOf(labelField)].toLowerCase();
        if (labelField === "attack_cat" && rawLabel) labels.push(rawLabel === "normal" ? "Normal" : "Attack");
        else if (["1", "1.0", "attack", "malicious"].includes(rawLabel)) labels.push("Attack");
        else if (["0", "normal", "benign"].includes(rawLabel)) labels.push("Normal");
        else errors.push(`Row ${rowIndex + 2}: ${labelField} must identify an attack or normal flow.`);
      }
    }
  });

  return { rows: errors.length ? [] : rows, labels: errors.length ? [] : labels, errors };
}

export function createPreset(kind: "normal" | "attack" | "mixed") {
  const count = kind === "mixed" ? 100 : 50;
  return Array.from({ length: count }, (_, index) => {
    const attack = kind === "attack" || (kind === "mixed" && index % 2 === 1);
    return {
      dur: attack ? 0.000011 + index / 1_000_000 : 0.42 + index / 100,
      proto: attack ? "udp" : "tcp",
      service: attack ? "dns" : "http",
      state: attack ? "INT" : "FIN",
      spkts: attack ? 2 : 12,
      dpkts: attack ? 0 : 10,
      sbytes: attack ? 114 + index : 1098 + index * 3,
      dbytes: attack ? 0 : 8224 + index * 7,
      rate: attack ? 90909.09 : 34.2,
      sttl: attack ? 254 : 62,
      dttl: attack ? 0 : 252,
      sload: attack ? 82909088 : 42118.4,
      dload: attack ? 0 : 312442.1,
      sloss: 0,
      dloss: attack ? 0 : 2,
      sinpkt: attack ? 0.011 : 31.2,
      dinpkt: attack ? 0 : 30.8,
      sjit: attack ? 0 : 128.4,
      djit: attack ? 0 : 92.1,
      swin: 255,
      stcpb: attack ? 0 : 1934821,
      dtcpb: attack ? 0 : 3291044,
      dwin: attack ? 0 : 255,
      tcprtt: attack ? 0 : 0.132,
      synack: attack ? 0 : 0.061,
      ackdat: attack ? 0 : 0.071,
      smean: attack ? 57 : 91,
      dmean: attack ? 0 : 822,
      trans_depth: attack ? 0 : 1,
      response_body_len: attack ? 0 : 4096,
      ct_srv_src: attack ? 17 : 3,
      ct_state_ttl: attack ? 2 : 1,
      ct_dst_ltm: attack ? 17 : 4,
      ct_src_dport_ltm: attack ? 17 : 2,
      ct_dst_sport_ltm: attack ? 17 : 1,
      ct_dst_src_ltm: attack ? 17 : 3,
      is_ftp_login: 0,
      ct_ftp_cmd: 0,
      ct_flw_http_mthd: attack ? 0 : 1,
      ct_src_ltm: attack ? 17 : 4,
      ct_srv_dst: attack ? 17 : 3,
      is_sm_ips_ports: 0,
      ground_truth: attack ? "Attack" : "Normal",
    } satisfies Record<string, string | number>;
  });
}
