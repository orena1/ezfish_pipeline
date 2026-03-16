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
    Find landmark file, preferring manual over auto.

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
    tuple: (path, source) where source is "manual", "auto", or None
    """
    base_dir = Path(base_dir)

    # Manual file (user-created)
    manual_name = f"{prefix}plane{plane}_to_HCR{hcr_ref}{suffix}"
    manual_path = base_dir / manual_name

    # Auto file (pipeline-generated)
    auto_name = f"{prefix}plane{plane}_to_HCR{hcr_ref}_auto{suffix}"
    auto_path = base_dir / auto_name

    if manual_path.exists():
        return manual_path, "manual"
    elif auto_path.exists():
        return auto_path, "auto"
    else:
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


def export_landmarks_to_bigwarp_csv(output_path, landmarks_df, pixel_spacing):
    """
    Export corrected landmarks to BigWarp CSV format.

    BigWarp format (8 columns, no header):
    name,enabled,src_x,src_y,src_z,dst_x,dst_y,dst_z

    Parameters
    ----------
    output_path : Path
        Output CSV path (should end with _auto.csv)
    landmarks_df : DataFrame
        Landmarks with columns: name, enabled, 2p_x, 2p_y, 2p_z, hcr_x, hcr_y, hcr_z
        HCR coords should already be corrected by registration shifts
    pixel_spacing : tuple
        (z_spacing, y_spacing, x_spacing) in microns for coordinate conversion
        
    """
    output_path = Path(output_path)

    # Warn if about to overwrite a manual file (should never happen with _auto suffix)
    if output_path.exists() and "_auto" not in output_path.name:
        rprint(f"[red]WARNING: Would overwrite manual file {output_path.name}![/red]")
        rprint("[red]Aborting to protect user data.[/red]")
        return None

    # Write CSV without header (BigWarp format)
    landmarks_df.to_csv(output_path, index=False, header=False)
    return output_path


def prompt_registration_checkpoint(qa_paths, auto_landmarks_path, step_name, plane_idx):
    """
    Interactive checkpoint after auto-registration.

    Returns
    -------
    str: "accept", "refine", or "skip"
    """
    rprint(f"\n[bold cyan]{'='*60}[/bold cyan]")
    rprint(f"[bold green]Plane {plane_idx}: Auto-{step_name} Complete[/bold green]")
    rprint(f"[bold cyan]{'='*60}[/bold cyan]\n")

    rprint("[bold]>>> STEP 1: Check the QA images in ImageJ/Fiji <<<[/bold]")
    for i, path in enumerate(qa_paths):
        label = "BEFORE" if i == 0 else "AFTER"
        rprint(f"  {label}: [blue]{path}[/blue]")

    rprint(f"\n[bold]>>> STEP 2: Refined landmarks saved to <<<[/bold]")
    rprint(f"  [yellow]{auto_landmarks_path}[/yellow]")

    rprint("\n[bold]After checking QA images, choose:[/bold]")
    rprint("  [green][y][/green] Accept - continue with auto results")
    rprint("  [yellow][r][/yellow] Refine - edit landmarks_auto.csv in BigWarp, then re-run")
    rprint("  [red][n][/red] Skip - skip this plane, continue with others\n")

    choice = Prompt.ask("Your choice", choices=["y", "r", "n"], default="y")

    if choice == "y":
        return "accept"
    elif choice == "r":
        rprint(f"\n[yellow]Edit the landmarks file, save it, then press Enter:[/yellow]")
        rprint(f"  {auto_landmarks_path}")
        input("Press Enter when ready...")
        return "refine"
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
