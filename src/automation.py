"""
Automation utilities for registration checkpoints and landmark management.
Keeps it simple: ~100 lines, no classes, reuse existing patterns.
"""

import pandas as pd
from pathlib import Path
from rich import print as rprint
from rich.prompt import Prompt


def find_landmark_file(base_dir, plane, prefix="", suffix="_landmarks.csv", hcr_ref="1"):
    """
    Find landmark file.

    Search order: expected prefix first, then alternate prefix
    (handles hires↔lowres mode switches without requiring renames).

    Parameters
    ----------
    base_dir : Path
        Directory containing landmark files
    plane : int
        Plane index
    prefix : str
        File prefix (e.g., "lowres_to_hires_" or "hires_stitched_")
    suffix : str
        File suffix (default: "_landmarks.csv")
    hcr_ref : str
        HCR reference round number (default: "1")

    Returns
    -------
    tuple: (path, source) where source is "manual" or None
    """
    base_dir = Path(base_dir)

    # Try expected prefix first
    manual_name = f"{prefix}plane{plane}_to_HCR{hcr_ref}{suffix}"
    manual_path = base_dir / manual_name

    if manual_path.exists():
        return manual_path, "manual"

    # Fallback: try alternate prefix (user may have switched hires↔lowres mode)
    alt_prefix = "" if prefix else "hires_stitched_"
    alt_manual = base_dir / f"{alt_prefix}plane{plane}_to_HCR{hcr_ref}{suffix}"

    if alt_manual.exists():
        rprint(f"[yellow]Note: using landmarks with '{alt_prefix}' prefix (expected '{prefix}' prefix for current mode)[/yellow]")
        return alt_manual, "manual"

    return None, None


def prompt_for_missing_file(file_path, file_description, instructions=None, tool_hint="BigWarp"):
    """
    Display message about missing file, wait for user to create it, re-check.
    Loops until file exists.

    Parameters
    ----------
    file_path : Path
        Expected file path
    file_description : str
        Human-readable description (e.g., "2P-to-HCR landmarks for Plane 3")
    instructions : str, optional
        Additional instructions for creating the file
    tool_hint : str
        Tool name to suggest (default: "BigWarp")

    Returns
    -------
    bool
        True when file exists
    """
    file_path = Path(file_path)

    while not file_path.exists():
        rprint(f"\n[bold cyan]{'='*60}[/bold cyan]")
        rprint(f"[bold yellow]Missing Required File[/bold yellow]")
        rprint(f"[bold cyan]{'='*60}[/bold cyan]\n")

        rprint(f"{file_description}")
        rprint(f"[bold]Expected path:[/bold]")
        rprint(f"  [yellow]{file_path}[/yellow]\n")

        if instructions:
            rprint(f"{instructions}\n")

        rprint(f"Create this file in {tool_hint}, then press [green]Enter[/green] to continue...")
        input()

        if file_path.exists():
            rprint(f"[green]Found: {file_path.name}[/green]\n")
            return True
        else:
            rprint(f"[red]File still not found. Please check the path and try again.[/red]")

    return True



def prompt_registration_checkpoint(qa_paths, step_name, plane_idx):
    """
    Interactive checkpoint after auto-registration.

    Returns
    -------
    str: "accept" or "skip"
    """
    rprint(f"\n[bold cyan]{'='*60}[/bold cyan]")
    rprint(f"[bold green]Plane {plane_idx}: Auto-{step_name} Complete[/bold green]")
    rprint(f"[bold cyan]{'='*60}[/bold cyan]\n")

    rprint("[bold]Check the QA images in ImageJ/Fiji:[/bold]")
    for i, path in enumerate(qa_paths):
        label = "BEFORE" if i == 0 else "AFTER"
        rprint(f"  {label}: [blue]{path}[/blue]")

    rprint("\n[bold]After checking QA images, choose:[/bold]")
    rprint("  [green][y][/green] Accept - continue with mask matching")
    rprint("  [red][n][/red] Skip - skip this plane\n")

    choice = Prompt.ask("Your choice", choices=["y", "n"], default="y")

    if choice == "y":
        return "accept"
    else:
        return "skip"


def print_plane_summary(results):
    """
    Print summary of plane processing results.

    Parameters
    ----------
    results : dict
        {plane_idx: "accepted" | "skipped" | "failed"}
    """
    rprint(f"\n[bold cyan]{'='*60}[/bold cyan]")
    rprint("[bold]Registration Summary[/bold]")
    rprint(f"[bold cyan]{'='*60}[/bold cyan]")

    accepted = [p for p, r in results.items() if r == "accepted"]
    skipped = [p for p, r in results.items() if r == "skipped"]
    failed = [p for p, r in results.items() if r == "failed"]

    if accepted:
        rprint(f"  [green]Accepted:[/green] planes {accepted}")
    if skipped:
        rprint(f"  [yellow]Skipped:[/yellow] planes {skipped}")
    if failed:
        rprint(f"  [red]Failed:[/red] planes {failed}")

    rprint("")
