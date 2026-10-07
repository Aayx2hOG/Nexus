import type {
  Alert,
  AlertDetail,
  AlertStats,
  FlowPrediction,
  Health,
  ModelManifest,
  ModelOption,
  ProbeResult,
  ReviewSample,
  ReviewStatus,
} from "./types";

const features = {
  dur: 0.000011,
  proto: "udp",
  service: "dns",
  state: "INT",
  spkts: 2,
  dpkts: 0,
  sbytes: 114,
  dbytes: 0,
  rate: 90909.09,
  sttl: 254,
  dttl: 0,
  sload: 82909088,
  dload: 0,
  sloss: 0,
  dloss: 0,
  sinpkt: 0.011,
  dinpkt: 0,
  sjit: 0,
  djit: 0,
  swin: 255,
  stcpb: 0,
  dtcpb: 0,
  dwin: 0,
  tcprtt: 0,
  synack: 0,
  ackdat: 0,
  smean: 57,
  dmean: 0,
  trans_depth: 0,
  response_body_len: 0,
  ct_srv_src: 17,
  ct_state_ttl: 2,
  ct_dst_ltm: 17,
  ct_src_dport_ltm: 17,
  ct_dst_sport_ltm: 17,
  ct_dst_src_ltm: 17,
  is_ftp_login: 0,
  ct_ftp_cmd: 0,
  ct_flw_http_mthd: 0,
  ct_src_ltm: 17,
  ct_srv_dst: 17,
  is_sm_ips_ports: 0,
};

export const mockAlerts: AlertDetail[] = [
  {
    id: "ALT-8F2C1A",
    flowId: "FLW-92D17E44",
    createdAt: "2025-02-18T14:32:08Z",
    severity: "Critical",
    source: "175.45.176.3:1043",
    destination: "149.171.126.14:53",
    score: 0.9417,
    threshold: 0.5776925765603604,
    category: "Generic",
    reviewStatus: "Pending Review",
    protocol: "udp",
    service: "dns",
    state: "INT",
    shap: [
      { name: "sttl", value: 254, contribution: 0.312 },
      { name: "ct_srv_src", value: 17, contribution: 0.184 },
      { name: "sbytes", value: 114, contribution: 0.092 },
      { name: "dur", value: 0.000011, contribution: -0.041 },
      { name: "dbytes", value: 0, contribution: 0.038 },
    ],
    features,
    reviewVersion: 3,
    reviews: [
      {
        timestamp: "2025-02-18T14:36:21Z",
        verdict: "Investigating",
        notes: "Source pattern requires DNS traffic review.",
        version: 2,
        reviewer: "soc-analyst-04",
      },
    ],
  },
  {
    id: "ALT-4B77D0",
    flowId: "FLW-23AC810B",
    createdAt: "2025-02-18T13:48:19Z",
    severity: "High",
    source: "59.166.0.8:49730",
    destination: "149.171.126.8:80",
    score: 0.8124,
    threshold: 0.5776925765603604,
    category: "Exploits",
    reviewStatus: "Investigating",
    protocol: "tcp",
    service: "http",
    state: "FIN",
    shap: [
      { name: "ct_srv_src", value: 11, contribution: 0.229 },
      { name: "sbytes", value: 1098, contribution: 0.116 },
      { name: "sttl", value: 62, contribution: 0.074 },
    ],
    features: { ...features, sbytes: 1098, dbytes: 8224, sttl: 62, ct_srv_src: 11 },
    reviewVersion: 1,
    reviews: [
      {
        timestamp: "2025-02-18T13:53:41Z",
        verdict: "Investigating",
        notes: "HTTP flow retained for payload and destination review.",
        version: 1,
        reviewer: "soc-analyst-07",
      },
    ],
  },
  {
    id: "ALT-11E9AF",
    flowId: "FLW-77AA031C",
    createdAt: "2025-02-18T12:11:54Z",
    severity: "Medium",
    source: "59.166.0.2:21804",
    destination: "149.171.126.5:25",
    score: 0.6321,
    threshold: 0.5776925765603604,
    category: "Reconnaissance",
    reviewStatus: "Confirmed Attack",
    protocol: "tcp",
    service: "smtp",
    state: "CON",
    shap: [
      { name: "dur", value: 0.42, contribution: 0.12 },
      { name: "ct_srv_src", value: 8, contribution: 0.077 },
    ],
    features: { ...features, dur: 0.42, ct_srv_src: 8 },
    reviewVersion: 2,
    reviews: [
      {
        timestamp: "2025-02-18T12:22:10Z",
        verdict: "Confirmed Attack",
        notes: "Repeated service enumeration confirmed.",
        version: 2,
      },
    ],
  },
  {
    id: "ALT-90C21B",
    flowId: "FLW-15E22CA1",
    createdAt: "2025-02-18T11:26:33Z",
    severity: "Low",
    source: "59.166.0.7:41220",
    destination: "149.171.126.2:443",
    score: 0.5912,
    threshold: 0.5776925765603604,
    category: "Generic",
    reviewStatus: "False Positive",
    protocol: "tcp",
    service: "http",
    state: "FIN",
    shap: [],
    features: { ...features, dur: 1.82, sbytes: 2210, dbytes: 18442, sttl: 62, ct_srv_src: 2 },
    reviewVersion: 2,
    reviews: [
      {
        timestamp: "2025-02-18T11:39:10Z",
        verdict: "False Positive",
        notes: "Expected internal service traffic verified against the flow record.",
        version: 2,
        reviewer: "soc-analyst-02",
      },
    ],
  },
];

const baseDemoAlerts = [...mockAlerts];
for (let index = 0; index < 8; index += 1) {
  const source = baseDemoAlerts[index % baseDemoAlerts.length];
  mockAlerts.push({
    ...source,
    id: `ALT-DMO${String(index + 1).padStart(3, "0")}`,
    flowId: `FLW-DMO${String(index + 1).padStart(5, "0")}`,
    createdAt: new Date(Date.parse(source.createdAt) - (index + 1) * 19 * 60_000).toISOString(),
    source: `10.20.${index + 4}.${20 + index}:${4100 + index}`,
    destination: `172.18.0.${12 + index}:${index % 2 ? 443 : 53}`,
  });
}

const pause = <T,>(data: T) =>
  new Promise<T>((resolve) => window.setTimeout(() => resolve(data), 260));

export const mockAdapter = {
  health: () =>
    pause<Health>({ system: "Operational", api: "Operational", model: "Operational" }),
  alerts: () => pause<Alert[]>(mockAlerts),
  alert: (id: string) => pause(mockAlerts.find((alert) => alert.id === id) ?? mockAlerts[0]),
  stats: () =>
    pause<AlertStats>({
      last24Hours: mockAlerts.length,
      awaitingReview: mockAlerts.filter((a) => a.reviewStatus === "Pending Review").length,
      confirmedAttacks: mockAlerts.filter((a) => a.reviewStatus === "Confirmed Attack").length,
      falsePositives: mockAlerts.filter((a) => a.reviewStatus === "False Positive").length,
      investigating: mockAlerts.filter((a) => a.reviewStatus === "Investigating").length,
      totalPredictions: 128,
    }),
  samples: () =>
    pause<ReviewSample[]>([
      {
        id: "FLW-40B2C6DE",
        ingestTime: "2025-02-18T11:42:31Z",
        protocol: "tcp",
        service: "ftp-data",
        state: "FIN",
        score: 0.5412,
        threshold: 0.5776925765603604,
        auditStatus: "Unreviewed",
        features: { ...features, dur: 1.246, sbytes: 2646, dbytes: 22558 },
      },
      {
        id: "FLW-1027AAB9",
        ingestTime: "2025-02-18T10:18:06Z",
        protocol: "udp",
        service: "dns",
        state: "INT",
        score: 0.4871,
        threshold: 0.5776925765603604,
        auditStatus: "Needs Triage",
        features: { ...features, ct_srv_src: 14, sttl: 60 },
      },
      {
        id: "FLW-98E21D4A",
        ingestTime: "2025-02-18T09:44:52Z",
        protocol: "tcp",
        service: "http",
        state: "FIN",
        score: 0.5621,
        threshold: 0.5776925765603604,
        auditStatus: "Missed Attack",
        features: { ...features, dur: 0.031, sbytes: 412, dbytes: 118, ct_srv_src: 15 },
      },
      {
        id: "FLW-A0617C22",
        ingestTime: "2025-02-18T08:29:14Z",
        protocol: "tcp",
        service: "smtp",
        state: "CON",
        score: 0.5538,
        threshold: 0.5776925765603604,
        auditStatus: "Malicious Intent",
        features: { ...features, dur: 0.19, sbytes: 782, dbytes: 214, ct_dst_ltm: 16 },
      },
    ]),
  probe: (target: string) => {
    let hash = 0;
    for (let i = 0; i < target.length; i++) {
      hash = (hash << 5) - hash + target.charCodeAt(i);
      hash |= 0;
    }
    const seed = Math.abs(hash);
    const dns = 12 + (seed % 45);
    const tcp = 18 + ((seed >> 2) % 65);
    const tls = 35 + ((seed >> 4) % 110);
    const ttfb = dns + tcp + tls + 40 + ((seed >> 3) % 120);
    const total = ttfb + 25 + ((seed >> 5) % 80);
    const attackLike = /attack|exploit|malicious|test-danger|bad/i.test(target);
    const score = attackLike ? 0.88 + ((seed % 100) / 1000) : 0.08 + ((seed % 350) / 1000);
    const threatLevel = attackLike || score > 0.57769 ? "High" : "Normal";
    const verdict = threatLevel === "Normal" ? "Normal Traffic" : "Threat Flagged";

    return pause<ProbeResult>({
      id: crypto.randomUUID(),
      target,
      timestamp: new Date().toISOString(),
      verdict,
      score,
      threshold: 0.5776925765603604,
      status: 200,
      threatLevel,
      measurements: {
        "DNS resolution": `${dns} ms`,
        "TCP connection": `${tcp} ms`,
        "TLS handshake": `${tls} ms`,
        "Time to first byte": `${ttfb} ms`,
        "Total request time": `${total} ms`,
        "HTTP version": seed % 2 === 0 ? "HTTP/2" : "HTTP/1.1",
        HTTPS: "Yes",
      },
      shap: [
        { name: "dur", value: +(total / 1000).toFixed(4), contribution: attackLike ? 0.312 : -0.146 },
        { name: "dbytes", value: 1200 + (seed % 28000), contribution: attackLike ? 0.184 : -0.084 },
      ],
    });
  },
  manifest: () =>
    pause<ModelManifest>({
      architecture: "LightGBM",
      bundleVersion: "nexus-lgbm-1.0.0",
      threshold: 0.5776925765603604,
      selection: {
        Accuracy: "95.5056%",
        Precision: "98.0079%",
        Recall: "95.2548%",
        F1: "96.6118%",
        "False positive rate": "3.9790%",
      },
      artifactHashes: {
        "model.joblib": "4915b64d918e9e1d7b8fb1ea535976c32c2b3ff89a7c01102894d9a10613f163",
        "preprocessor.joblib":
          "170cd5034dd4e2f1fc85b11dd35a30b88ec85c7259710ca91e2d56f11a14fd8d",
      },
      sourceHashes: {
        "feature_schema.json":
          "dde6cc2950127fcc226aed94cfe4ea87f655a624de198f4d6f4997fa52ce4b0f",
      },
    }),
  models: () =>
    pause<ModelOption[]>([
      {
        id: "uncertainty_band_ae10",
        name: "Uncertainty-band four-mode, AE 10% attack oriented",
        description:
          "LightGBM + Baseline Autoencoder with 10% FPR anomaly boundaries. Focuses on attack recovery in the uncertainty band [0.10, 0.65].",
        architecture: "LightGBM + Baseline AE (4-Mode)",
        decision_threshold: 0.5776925765603604,
        focus: "Attack Oriented (Higher Recall)",
        ae_budget: "10% FPR",
        is_fusion: true,
      },
      {
        id: "mode_confidence_ae05",
        name: "Mode confidence, AE 5%, soc oriented",
        description:
          "LightGBM + Baseline Autoencoder with 5% FPR anomaly boundaries. Focuses on SOC triage and reducing false alarms by 992 cases.",
        architecture: "LightGBM + Baseline AE (Confidence Mode)",
        decision_threshold: 0.5776925765603604,
        focus: "SOC Oriented (Lower False Positives)",
        ae_budget: "5% FPR",
        is_fusion: true,
      },
    ]),
  predict: (flows: Record<string, string | number>[], modelId: string = "uncertainty_band_ae10") =>
    pause<FlowPrediction[]>(
      flows.map((flow, index) => {
        const isGroundTruthAttack =
          flow.ground_truth === "Attack" || flow.label === 1 || flow.label === "1";
        const isGroundTruthNormal =
          flow.ground_truth === "Normal" || flow.label === 0 || flow.label === "0";
        const label = isGroundTruthAttack
          ? "Attack"
          : isGroundTruthNormal
          ? "Normal"
          : undefined;

        const sttl = Number(flow.sttl) || 0;
        const ctSrv = Number(flow.ct_srv_src) || 0;
        const sbytes = Number(flow.sbytes) || 0;
        const dbytes = Number(flow.dbytes) || 0;

        // Base supervised signal simulation
        let score: number;
        if (isGroundTruthAttack) {
          // Attacks range from obvious (0.80+) to borderline (0.35-0.55)
          const isBorderline = index % 5 === 0;
          score = isBorderline ? 0.42 + (index % 15) / 100 : 0.78 + (index % 18) / 100;
        } else if (isGroundTruthNormal) {
          // Normal traffic ranges from clean (0.05-0.25) to noisy/ambiguous (0.45-0.58)
          const isAmbiguous = index % 8 === 0;
          score = isAmbiguous ? 0.46 + (index % 12) / 100 : 0.08 + (index % 25) / 100;
        } else {
          // Unlabeled flow: compute score from network heuristics
          const suspicious = sttl > 200 || ctSrv > 14 || (sbytes > 2000 && dbytes === 0);
          score = suspicious ? 0.76 + (index % 20) / 100 : 0.12 + (index % 30) / 100;
        }
        score = Math.min(0.999, Math.max(0.001, +score.toFixed(4)));

        // Autoencoder reconstruction error simulation
        let aeError: number;
        let aeMode: string;
        let isAlert: boolean;
        const defaultThreshold = 0.5776925765603604;

        if (modelId === "uncertainty_band_ae10") {
          // Model 1: Uncertainty-band four-mode, AE 10% attack oriented
          // Lower anomaly threshold (10% budget), recovers borderline attacks in [0.10, 0.65]
          aeError = +(0.015 + ((score * 0.05) + (index % 7) * 0.006)).toFixed(4);
          if (aeError >= 0.045) aeMode = "high";
          else if (aeError >= 0.030) aeMode = "mid";
          else if (aeError >= 0.018) aeMode = "low";
          else aeMode = "no_anomaly";

          if (score >= defaultThreshold) {
            isAlert = true;
          } else if (score >= 0.10 && score <= 0.65 && (aeMode === "high" || aeMode === "mid")) {
            // Recover borderline attack via Autoencoder anomaly mode
            isAlert = true;
          } else {
            isAlert = false;
          }
        } else {
          // Model 2: Mode confidence, AE 5%, soc oriented
          // Higher anomaly threshold (5% budget), strict confidence cutoff to eliminate false alarms
          aeError = +(0.010 + ((score * 0.035) + (index % 6) * 0.004)).toFixed(4);
          if (aeError >= 0.055) aeMode = "high";
          else if (aeError >= 0.038) aeMode = "mid";
          else if (aeError >= 0.022) aeMode = "low";
          else aeMode = "no_anomaly";

          // Prune false positives: requires solid score and AE agreement
          if (score >= 0.62) {
            isAlert = true;
          } else if (score >= defaultThreshold && (aeMode === "high" || aeMode === "mid")) {
            isAlert = true;
          } else {
            // Borderline / noisy flows pruned to protect SOC analyst queues
            isAlert = false;
          }
        }

        return {
          flowId: `FLW-PRESET-${String(index + 1).padStart(4, "0")}`,
          alertId: isAlert ? mockAlerts[index % mockAlerts.length].id : undefined,
          score,
          threshold: defaultThreshold,
          decision: isAlert ? "Alert" : "Normal",
          category: isAlert ? "Generic" : undefined,
          groundTruth: label,
          features: Object.fromEntries(
            Object.entries(flow).filter(([key]) => key !== "ground_truth" && key !== "label")
          ),
          aeMode,
          aeError,
          modelId,
        };
      }),
    ),
  submitAlertFeedback: (id: string, data: { verdict: ReviewStatus; notes?: string }) => {
    const alert = mockAlerts.find((a) => a.id === id);
    if (alert) {
      alert.reviewStatus = data.verdict;
      alert.reviewVersion = (alert.reviewVersion || 1) + 1;
      alert.reviews.push({
        timestamp: new Date().toISOString(),
        verdict: data.verdict as Exclude<ReviewStatus, "Pending Review">,
        notes: data.notes?.trim() || "Recorded decision.",
        version: alert.reviewVersion,
        reviewer: "soc-analyst-01",
      });
    }
    return pause({ ok: true });
  },
};
