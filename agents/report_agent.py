"""Stage 5 [Report (AI)]: Plain-English suggestion brief for manual human review.

Runs ONLY after [4. Risk Gate] approves an idea. Includes candid model-uncertainty
framing, structured catalysts/risks, and an explicit "What Would Prove It Wrong"
invalidation section. Strictly adheres to the core rule:
Agents only suggest. Humans decide and trade manually.
"""
from dataclasses import dataclass
from pathlib import Path
from core.llm import LLMConfig, generate_text


def _build_deterministic_markdown(
    ticker: str,
    market: dict,
    research: dict,
    strategy_idea: dict,
    risk_gate: dict,
    monitor_precheck: dict,
    bankroll: float,
) -> str:
    price = float(strategy_idea['entry_reference_price'])
    shares = int(risk_gate['shares'])
    notional = float(risk_gate['notional'])
    stop_price = float(risk_gate['stop_price'])
    target_price = float(risk_gate['target_price'])
    nom_risk = float(risk_gate['nominal_risk'])
    horizon_days = int(strategy_idea.get('horizon_days', 5))
    alloc_pct = (notional / bankroll * 100.0) if bankroll > 0 else 0.0
    risk_pct = (nom_risk / bankroll * 100.0) if bankroll > 0 else 0.0

    rationale_bullets = '\n'.join(f"- {item}" for item in strategy_idea.get('rationale', []))
    catalyst_bullets = '\n'.join(f"- {item}" for item in research.get('catalysts', []))
    risk_bullets = '\n'.join(f"- {item}" for item in research.get('risks', []))

    event_flags = research.get('event_flags') or []
    flag_bullets = (
        '\n'.join(f"- **[{f['severity'].upper()} · {f['kind']}]** {f['summary']}" for f in event_flags[:4])
        if event_flags
        else '- No high-severity adverse news events flagged in recent headlines.'
    )

    ml_info = strategy_idea.get('ml_forecast')
    if isinstance(ml_info, dict) and ml_info.get('available'):
        ml_line = (
            f"- **Walk-Forward {ml_info['horizon_days']}d ML Check:** P(up) = **{ml_info['p_up']:.0%}** "
            f"(held-out accuracy **{ml_info['backtest_accuracy']:.0%}** over last {ml_info['test_window']} settled bars). "
            f"*Be candid about uncertainty: a 55–60% model is only modestly better than a coin flip, so position caps and stop discipline carry the real protection.*\n"
        )
    else:
        ml_line = ''

    monitor_warnings = monitor_precheck.get('warnings', [])
    monitor_section = (
        '\n'.join(f"- **Monitor note:** {w}" for w in monitor_warnings)
        if monitor_warnings
        else '- All upstream agent checks passed with zero cross-agent warnings.'
    )

    return (
        f"### Paper Trade Suggestion: {ticker} ({strategy_idea['strategy_name']})\n\n"
        f"**Plain-English Summary:** Based on real daily closing data through **{market['as_of']}**, "
        f"**{ticker}** passed the deterministic Risk Gate as a paper long suggestion. "
        f"At the latest close of **${price:,.2f}**, your risk profile permits up to **{shares} whole share(s)** "
        f"(**${notional:,.2f}**, or **{alloc_pct:.1f}%** of your **${bankroll:,.2f}** paper capital) with a "
        f"protective stop near **${stop_price:,.2f}** (**${nom_risk:,.2f}** / **{risk_pct:.2f}%** nominal risk).\n\n"
        f"#### Why This Idea Surfaced (Market + Research + Strategy)\n"
        f"{rationale_bullets}\n"
        f"{ml_line}"
        f"- **Research Summary:** {research.get('summary', '')}\n\n"
        f"#### Supporting Catalysts & Flagged Headlines\n"
        f"{catalyst_bullets}\n"
        f"{flag_bullets}\n\n"
        f"#### Risk-Gated Position Plan (Paper / Manual Reference)\n"
        f"- **Reference Entry (last daily close):** ${price:,.2f} *(evaluate against next session's live open)*\n"
        f"- **Suggested Max Size:** {shares} whole share(s) (${notional:,.2f} notional; cap {risk_gate['max_allocation_pct']:g}%)\n"
        f"- **Protective Stop-Loss ({risk_gate['stop_loss_pct']:g}%):** ${stop_price:,.2f} "
        f"(nominal loss ${nom_risk:,.2f}; cap {risk_gate['max_risk_pct']:g}%)\n"
        f"- **Illustrative 2R Reference Target:** ${target_price:,.2f} (reward/risk ~{risk_gate['reward_risk_ratio']}x)\n\n"
        f"#### What Would Prove It Wrong\n"
        f"- A daily close or intraday drop breaching the protective stop at **${stop_price:,.2f}**.\n"
        f"- Failure to hold constructive trend/momentum over the next **{horizon_days} trading days** "
        f"or an exit signal from **{strategy_idea['strategy_name']}**.\n"
        f"- Any sudden high-severity adverse event (earnings miss, guidance cut, regulatory probe, or litigation).\n\n"
        f"#### Key Risks & Monitor Agent Checks\n"
        f"{risk_bullets}\n"
        f"{monitor_section}\n"
        f"- **Gap Disclaimer:** {risk_gate['warning']}\n\n"
        f"---\n"
        f"**Human Decision Required:** *Agents only suggest — humans decide and trade manually.* "
        f"No order has been or will be placed by this software. Review live quotes independently and choose "
        f"below whether to log a paper approval or skip this idea."
    )


@dataclass
class ReportAgent:
    llm_config: LLMConfig | None = None

    def generate(
        self,
        ticker: str,
        market: dict,
        research: dict,
        strategy_idea: dict,
        risk_gate: dict,
        monitor_precheck: dict,
        bankroll: float,
        output_dir: Path | None = None,
    ) -> dict:
        if not risk_gate.get('passed'):
            raise ValueError('ReportAgent must only run for ideas that pass the Risk Gate')

        fallback_md = _build_deterministic_markdown(
            ticker=ticker,
            market=market,
            research=research,
            strategy_idea=strategy_idea,
            risk_gate=risk_gate,
            monitor_precheck=monitor_precheck,
            bankroll=bankroll,
        )

        system_prompt = (
            'You are the Report (AI) Agent in a paper-only stock suggestion workflow. '
            'NON-NEGOTIABLE RULE: Agents only suggest. Humans decide and trade manually. '
            'Use ONLY the facts provided — never invent news, prices, or numbers, never claim an order was placed, '
            'and never promise profits. Be candid about uncertainty (a 55-60% model is only slightly better than a coin flip). '
            'Include sections covering: The Idea, Why the Strategy & Model Think So, Risk-Gated Position Plan '
            '(with exact shares, notional, stop-loss price, and nominal dollar risk), Risks & Recent News, and '
            'What Would Prove It Wrong. End with an explicit reminder that the human decides manually.'
        )
        user_prompt = (
            f"Ticker: {ticker}\n"
            f"Paper Capital: ${bankroll:,.2f}\n"
            f"Market Data: {market}\n"
            f"Research Brief: { {k: v for k, v in research.items() if k != 'headlines'} }\n"
            f"Strategy Idea: { {k: v for k, v in strategy_idea.items() if k != 'strategy_votes'} }\n"
            f"Risk Gate Sizing: {risk_gate}\n"
            f"Monitor Agent Notes: {monitor_precheck.get('warnings', [])}\n"
        )
        llm_md, ai_mode = generate_text(system_prompt, user_prompt, self.llm_config)

        if llm_md and str(risk_gate['shares']) in llm_md:
            final_md = (
                llm_md.strip()
                + '\n\n---\n**Human Decision Required:** *Agents only suggest — humans decide and trade manually.* '
                f"Stop-loss reference: ${risk_gate['stop_price']:,.2f} | Nominal risk: ${risk_gate['nominal_risk']:,.2f}."
            )
        else:
            final_md = fallback_md

        headline = (
            f"Suggest Paper Long {ticker}: up to {risk_gate['shares']} share(s) "
            f"(${risk_gate['notional']:,.2f}) | Stop ${risk_gate['stop_price']:,.2f} "
            f"(${risk_gate['nominal_risk']:,.2f} nominal risk)"
        )

        saved_path: str | None = None
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            note_file = output_dir / f"{market.get('as_of', 'latest')}_{ticker.strip().upper()}.md"
            note_file.write_text(final_md, encoding='utf-8')
            saved_path = str(note_file)

        return {
            'ticker': ticker,
            'headline': headline,
            'markdown_report': final_md,
            'ai_mode': ai_mode,
            'saved_note_path': saved_path,
            'manual_only_notice': 'Agents only suggest. Humans decide and trade manually.',
        }
