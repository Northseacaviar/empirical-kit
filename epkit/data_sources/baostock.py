"""baostock 取数：A 股日线与指数，落到本地 csv。

为什么要单独做：spec 的 data.prices 要的是"每只股票一个 csv，含 date/close"。
手工从 baostock 拉数再对齐格式，是每个课题都要重做一遍的活，而且复权口径
很容易前后不一致。

口径：
- 默认 adjustflag=2（前复权）——事件研究的收益率必须连续，不复权价格在除权日
  会凭空跳水，直接污染窗口内的异常收益。
- 同时保留 pctChg（baostock 原始涨跌幅，单位 %）与 ret（用复权收盘价算的
  简单收益率，单位小数），存两列不是冗余：前者可与行情软件核对，后者才是模型输入。
- 交易日与停牌：tradestatus=0 的行保留，但 close 为空时不会有收益，由上游决定怎么处理。
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

FIELDS = "date,code,open,high,low,close,preclose,volume,amount,tradestatus,pctChg"


def _bs():
    try:
        import baostock as bs
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "未安装 baostock：uv pip install --python .venv/Scripts/python.exe baostock"
        ) from exc
    return bs


def _to_bs_code(code: str) -> str:
    """把 600000 / 000001 / sh.600000 统一成 baostock 的 sh.600000 形式。"""
    c = str(code).strip().lower()
    if "." in c:
        return c
    if c.startswith(("6", "9")):
        return f"sh.{c}"
    if c.startswith(("0", "3")):
        return f"sz.{c}"
    if c.startswith(("4", "8")):
        return f"bj.{c}"
    return f"sh.{c}"


def fetch_one(bs, code: str, start: str, end: str, adjustflag: str = "2") -> pd.DataFrame:
    rs = bs.query_history_k_data_plus(_to_bs_code(code), FIELDS, start_date=start,
                                      end_date=end, frequency="d", adjustflag=adjustflag)
    if rs.error_code != "0":
        raise RuntimeError(f"{code} 取数失败：{rs.error_code} {rs.error_msg}")
    rows = []
    while rs.next():
        rows.append(rs.get_row_data())
    df = pd.DataFrame(rows, columns=rs.fields)
    if df.empty:
        return df
    for c in ("open", "high", "low", "close", "preclose", "volume", "amount", "pctChg"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["tradestatus"] = pd.to_numeric(df["tradestatus"], errors="coerce").fillna(1).astype(int)
    df = df.sort_values("date").reset_index(drop=True)
    # 存成纯 6 位代码：CSMAR 与横截面表用的也是这个形态，后面合并才对得上
    df["code"] = (df["code"].str.split(".", n=1).str[1]
                  if df["code"].str.contains(r"\.").any() else df["code"])
    # ret 用复权价格算：前复权序列在除权日连续，pctChg 在不复权口径下有跳空
    df["ret"] = df["close"].pct_change()
    df.loc[df["tradestatus"] == 0, "ret"] = pd.NA
    return df


def fetch(codes: list[str], start: str, end: str, out_dir: str | Path,
          adjustflag: str = "2", force: bool = False, verbose: bool = True) -> dict:
    """拉一批代码，每只写一个 csv。已存在且不 force 的跳过（增量更新靠重跑 force）。"""
    bs = _bs()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    lg = bs.login()
    if lg.error_code != "0":
        raise RuntimeError(f"baostock 登录失败：{lg.error_code} {lg.error_msg}")

    done, skipped, failed = [], [], []
    try:
        for i, code in enumerate(codes, 1):
            path = out / f"{_to_bs_code(code).split('.')[1]}.csv"
            if path.exists() and not force:
                skipped.append(code)
                continue
            try:
                df = fetch_one(bs, code, start, end, adjustflag)
                if df.empty:
                    failed.append((code, "无数据（停牌/退市/代码错误）"))
                    continue
                df.to_csv(path, index=False, encoding="utf-8-sig")
                done.append((code, len(df)))
                if verbose:
                    print(f"[{i}/{len(codes)}] {code} → {path.name}（{len(df)} 行）")
            except Exception as exc:
                failed.append((code, f"{type(exc).__name__}: {exc}"))
            time.sleep(0.05)
    finally:
        bs.logout()
    return {"written": done, "skipped": skipped, "failed": failed, "out_dir": str(out)}


def fetch_index(codes: list[str], start: str, end: str, out_csv: str | Path,
                verbose: bool = True) -> Path:
    """指数行情写成一个长表：spec 的 data.index 要的就是长表。"""
    bs = _bs()
    lg = bs.login()
    if lg.error_code != "0":
        raise RuntimeError(f"baostock 登录失败：{lg.error_code} {lg.error_msg}")
    frames = []
    try:
        for code in codes:
            df = fetch_one(bs, code, start, end, adjustflag="3")
            if df.empty:
                continue
            df = df[df["close"].notna()].copy()
            df["bench"] = _to_bs_code(code)
            frames.append(df[["date", "code", "close", "ret", "bench"]])
            if verbose:
                print(f"指数 {code}：{len(df)} 行")
    finally:
        bs.logout()
    if not frames:
        raise RuntimeError("没有任何指数数据取到")
    out = Path(out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_csv(out, index=False, encoding="utf-8-sig")
    return out
