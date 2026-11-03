const fs = require("fs");
const path = require("path");
const { execFileSync } = require("child_process");
const solc = require("solc");
const ganache = require("ganache");
const { ethers } = require("ethers");

const ROOT = path.resolve(__dirname, "..");
const PY = process.env.PYTHON || "python";
const CLI = path.join(ROOT, "code", "argus", "attest_cli.py");
const N = 5, START = 2800, WEEK = 2805, LEVEL = 2;

function py(args) {
  return JSON.parse(execFileSync(PY, [CLI, ...args], { maxBuffer: 1 << 26 }).toString());
}

function compile() {
  const src = fs.readFileSync(path.join(__dirname, "AlarmOracle.sol"), "utf8");
  const input = {
    language: "Solidity",
    sources: { "AlarmOracle.sol": { content: src } },
    settings: { optimizer: { enabled: true, runs: 200 }, viaIR: true, evmVersion: "paris",
                outputSelection: { "*": { "*": ["abi", "evm.bytecode.object"] } } },
  };
  const out = JSON.parse(solc.compile(JSON.stringify(input)));
  for (const e of out.errors || []) if (e.severity === "error") throw new Error(e.formattedMessage);
  const c = out.contracts["AlarmOracle.sol"].AlarmOracle;
  return { abi: c.abi, bytecode: "0x" + c.evm.bytecode.object };
}

async function expectRevert(p, label) {
  try { await (await p).wait(); console.log(`FAIL ${label}: did not revert`); return false; }
  catch (e) { console.log(`ok   ${label}: reverted`); return true; }
}

(async () => {
  const { abi, bytecode } = compile();
  const provider = new ethers.BrowserProvider(ganache.provider({
    logging: { quiet: true }, chain: { chainId: 1337, hardfork: "shanghai" },
    miner: { blockGasLimit: 60_000_000 }, wallet: { totalAccounts: 2, defaultBalance: 1000 } }));
  const signer = await provider.getSigner(0);
  const { roots } = py(["roots", "--n", `${N}`, "--start-week", `${START}`]);
  const results = { n: N, deploy: null, attest: {}, negative: {} };

  for (const k of [1, 2, 3, 4, 5]) {
    const f = new ethers.ContractFactory(abi, bytecode, signer);
    const c = await f.deploy(roots, k, START);
    const rcpt = await c.deploymentTransaction().wait();
    if (k === 3) results.deploy = Number(rcpt.gasUsed);
    const addr = await c.getAddress();
    const digest = ethers.keccak256(ethers.toUtf8Bytes("weekly evidence"));
    const s = py(["sign", "--n", `${N}`, "--start-week", `${START}`, "--chain-id", "1337",
                  "--contract", addr, "--week", `${WEEK}`, "--level", `${LEVEL}`, "--digest", digest]);
    const onchain = await c.statement(WEEK, LEVEL, digest);
    if (onchain !== s.statement) throw new Error("statement mismatch");
    const sigs = s.sigs.slice(0, k).map(x => [x.op, x.sig, x.path]);
    const tx = await c.submitAttestation(WEEK, LEVEL, digest, sigs, { gasLimit: 50_000_000 });
    const r = await tx.wait();
    results.attest[k] = Number(r.gasUsed);
    console.log(`k=${k}  attestation gas ${r.gasUsed}  level=${await c.threatLevel()}`);

    if (k === 3) {
      const bad = s.sigs.slice(0, 2).map(x => [x.op, x.sig, x.path]);
      results.negative.below_quorum = await expectRevert(
        c.submitAttestation(WEEK + 1, LEVEL, digest, bad, { gasLimit: 50_000_000 }), "below quorum (wrong week for keys)");
      results.negative.reuse = await expectRevert(
        c.submitAttestation(WEEK, LEVEL, digest, sigs, { gasLimit: 50_000_000 }), "one-time key reuse");
      const dup = [sigs[0], sigs[0], sigs[1]];
      results.negative.duplicate = await expectRevert(
        c.submitAttestation(WEEK, LEVEL, digest, dup, { gasLimit: 50_000_000 }), "duplicate operator");
    }
  }

  {
    const f = new ethers.ContractFactory(abi, bytecode, signer);
    const c = await f.deploy(roots, 3, START); await c.deploymentTransaction().wait();
    const addr = await c.getAddress();
    const digest = ethers.keccak256(ethers.toUtf8Bytes("weekly evidence"));
    const s = py(["sign", "--n", `${N}`, "--start-week", `${START}`, "--chain-id", "1337",
                  "--contract", addr, "--week", `${WEEK}`, "--level", `${LEVEL}`, "--digest", digest]);
    const sigs = s.sigs.slice(0, 3).map(x => [x.op, x.sig, x.path]);
    const other = ethers.keccak256(ethers.toUtf8Bytes("forged evidence"));
    results.negative.tampered = await expectRevert(
      c.submitAttestation(WEEK, LEVEL, other, sigs, { gasLimit: 50_000_000 }), "tampered digest");
  }
  fs.mkdirSync(path.join(ROOT, "results"), { recursive: true });
  fs.writeFileSync(path.join(ROOT, "results", "alarm_oracle_gas.json"), JSON.stringify(results, null, 2));
  console.log(JSON.stringify(results));
})().catch(e => { console.error(e); process.exit(1); });
