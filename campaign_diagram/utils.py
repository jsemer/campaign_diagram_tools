## Parameter file extraction
import altair as alt
import pandas as pd
import re
from pathlib import Path
import itertools

from campaign_diagram.kernel_color import *
from campaign_diagram.debug import *
from campaign_diagram.kernel import Kernel
from campaign_diagram.cascade import Cascade
from campaign_diagram.fusion import FusionType

import pandas as pd

import numpy as np

GIGA = 10**9
GIGAB = 2**30

# -----------------------------------------------
# Extract B, I, and fusion type from filename
# -----------------------------------------------
# Assumes the CSV filename system we're using
def extract_params_from_filename(filename):
    pattern = r"B(\d+)_I(\d+)_results_([^_]+)_([^.]+)"
    match = re.search(pattern, filename)
    if match:
        B = int(match.group(1))
        I = int(match.group(2))
        accel = match.group(3)
        fusion_type = match.group(4)
        return B, I, accel, fusion_type
    return None, None, None, "Unknown"

# -----------------------------------------------
# Clean + aggregate per fusion group from CSV
# -----------------------------------------------
def transform_mambalaya_csv(csv_path, group_by="einsum", bandwidth_gbps=2039):
    filename = Path(csv_path).name
    B, I, accel, fusion_type = extract_params_from_filename(filename)

    df = pd.read_csv(csv_path)
    df = df[df["einsum"] != "total"].copy()

    grouped=df
    # grouped = df.groupby(group_by).agg({
    #     "total_traffic": "sum",
    #     "final_latency": "sum",
    #     "mem_latency": "sum",
    #     "comp_latency": "sum"
    # }).reset_index()

    grouped["Memory_Utilization"] = (grouped["total_traffic"] / grouped["final_latency"]) / (bandwidth_gbps * GIGA)

    ## We can't do this b/c it doesn't account for the two PE arrays - it returns relative util for either 2D OR 1D not both combined
    # grouped["Compute_Utilization"] = grouped["comp_latency"] / grouped["final_latency"]
    grouped["Compute_Utilization"] = grouped["compute_util"]

    grouped["mem_flag"] = ~np.isclose(grouped["Memory_Utilization"], grouped["mem_util"])
    grouped["comp_flag"] = ~np.isclose(grouped["Compute_Utilization"], grouped["compute_util"])
    
    current_time = 0.0
    grouped["Starting_Time"] = 0.0
    grouped["Time_Stamp"] = 0.0
    grouped["End_Time"] = 0.0
    for idx, row in grouped.iterrows():
        duration = row["final_latency"]
        grouped.at[idx, "Starting_Time"] = current_time
        grouped.at[idx, "Time_Stamp"] = current_time
        current_time += duration
        grouped.at[idx, "End_Time"] = current_time

    grouped = grouped.rename(columns={
        group_by: "Einsum",
        "final_latency": "runtime"
    })

    title = f"{fusion_type.replace('-', ' ').replace('sol', 'SOL').title()}, B={B}, I={I}"
    return grouped, grouped["Einsum"].tolist(), title


def collect_all_einsums(csv_dir, group_by="fusion_group"):
    pe_path = Path(csv_dir)
    csv_files = [f for f in pe_path.glob("*.csv") if "by_fusion_group" not in f.name]

    einsum_set = set()
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            einsums = df[df["einsum"] != "total"][group_by].unique()
            einsum_set.update(einsums)
        except Exception as e:
            print(f"Error processing {f.name}: {e}")
    return sorted(einsum_set)



# -----------------------------------------------
# Consistent color scale for Einsums
# -----------------------------------------------
def generate_color_scale(einsum_list, base_colors=None):
    if not base_colors:
      base_colors = [
          "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
          "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
          "#bcbd22", "#17becf"
      ]
    extended_palette = list(itertools.islice(itertools.cycle(base_colors), len(einsum_list)))
    return alt.Scale(domain=einsum_list, range=extended_palette)

def gen_fusion_group_boundaries(equal_df):
    # Sort just in case, then get the last row of each fusion group
    fusion_end_df = equal_df.sort_values("Starting_Time").groupby("fusion_group").tail(1)

    # Create vertical lines at each fusion group boundary
    return alt.Chart(fusion_end_df).mark_rule(
        color="gray", strokeDash=[3, 2], size=1
    ).encode(
        x=alt.X("End_Time:Q"),
        tooltip=["fusion_group", "End_Time"]
    )


def gen_campaign_block(equal_df, einsum_order, color_groups="fusion_group",
                       title="Campaign Diagram", color_scale=None, groups_color_scale=None,
                       gen_boundaries=True):
    
    shared_tooltip = ['Einsum', 'runtime', 'Memory_Utilization', 'Compute_Utilization', 'fusion_group']

    base = alt.Chart(equal_df).mark_line().encode(
        x=alt.X('Time_Stamp:Q', title="Time (ms)"),
        y=alt.Y('Compute_Utilization:Q', title="Compute Utilization")  # <-- fixed here
    ).transform_calculate(
        compute_limit_top="datum.Compute_Utilization+0.25",
        compute_limit_bot="datum.Compute_Utilization-0.25",
        y1_temp="datum.Memory_Utilization/4",
        y1="datum.y1_temp + datum.Compute_Utilization",
        y2="-datum.y1_temp + datum.Compute_Utilization"
    )

    tick = base.mark_rule(size=4).encode(
        x2="End_Time:Q",
        y=alt.Y('Compute_Utilization:Q', title="Compute Utilization"),  # <-- fixed here
        color=alt.Color('Einsum:N', sort=einsum_order, scale=color_scale, title="Einsum"),
        tooltip=shared_tooltip
    )

    mem_area = base.mark_rect(opacity=0.4).encode(
        x2="End_Time:Q",
        y=alt.Y('y1:Q', title="Compute Utilization"),  # <-- added for consistency
        y2=alt.Y2('y2:Q'),
        tooltip=shared_tooltip,
        color=alt.Color('Einsum:N', sort=einsum_order, scale=color_scale, title="Einsum")
    )

    guard_min = base.mark_rule(color='black', strokeDash=[1, 1], size=1, interpolate='step-after').encode(
        x2="End_Time:Q",
        y=alt.Y('compute_limit_top:Q', title="Compute Utilization"),  
        tooltip=shared_tooltip
        #color=alt.Color(f"{color_groups}:N", scale=groups_color_scale, title="Fusion Groups")
    )

    guard_max = base.mark_rule(color='black', size=1, strokeDash=[1, 1], interpolate='step-after').encode(
        x2="End_Time:Q",
        y=alt.Y('compute_limit_bot:Q', title="Compute Utilization"),
        tooltip=shared_tooltip
        #color=alt.Color(f"{color_groups}:N", scale=groups_color_scale, title="Fusion Groups")
    )

    final_chart = guard_min + guard_max + mem_area + tick 

    if gen_boundaries:
      final_chart +=  gen_fusion_group_boundaries(equal_df)

    return final_chart.properties(
        width=600,
        height=300,
        title=title
    )

# MP-style
def gen_campaign_block_shaded_with_dual_labels(df, einsum_order, title="Shaded Utilization with Axis Labels", 
                                               color_scale=None, gen_boundaries=True):
    shared_tooltip = ['Einsum', 'runtime', 'Memory_Utilization', 'Compute_Utilization', 'fusion_group']

    # === Compute shaded area (positive side)
    compute_area = alt.Chart(df).mark_rect(opacity=0.5).encode(
        x=alt.X('Starting_Time:Q', title="Time (ms)"),
        x2='End_Time:Q',
        y=alt.Y('zero:Q', title="Utilization", scale=alt.Scale(domain=[-1, 1])),
        y2='Compute_Utilization:Q',
        color=alt.Color('Einsum:N', sort=einsum_order, scale=color_scale),
        tooltip=shared_tooltip
    ).transform_calculate(
        zero='0'
    )

    # === Memory shaded area (negative side)
    memory_area = alt.Chart(df).mark_rect(opacity=0.5).encode(
        x='Starting_Time:Q',
        x2='End_Time:Q',
        y='zero:Q',
        y2='neg_mem_util:Q',
        color=alt.Color('Einsum:N', sort=einsum_order, scale=color_scale),
        tooltip=shared_tooltip
    ).transform_calculate(
        zero='0',
        neg_mem_util='-datum.Memory_Utilization'
    )

    # === Compute and memory horizontal lines
    compute_lines = alt.Chart(df).mark_rule(size=2).encode(
        x='Starting_Time:Q',
        x2='End_Time:Q',
        y='Compute_Utilization:Q',
        color=alt.Color('Einsum:N', sort=einsum_order, scale=color_scale),
        tooltip=shared_tooltip
    )

    memory_lines = alt.Chart(df).mark_rule(size=2).encode(
        x='Starting_Time:Q',
        x2='End_Time:Q',
        y='neg_mem_util:Q',
        color=alt.Color('Einsum:N', sort=einsum_order, scale=color_scale),
        tooltip=shared_tooltip
    ).transform_calculate(
        neg_mem_util='-datum.Memory_Utilization'
    )

    # === Horizontal zero line
    zero_line = alt.Chart(pd.DataFrame({'y': [0]})).mark_rule(
        color='black', size=1
    ).encode(
        y='y:Q'
    )

    # === Proper label layer with per-row dy encoding
    label_df = pd.DataFrame({
        'y': [0.95, -0.95],
        'label': ['Compute Utilization', 'Memory Utilization']
    })

    axis_labels = alt.Chart(label_df).mark_text(
        fontSize=18,
        fontWeight='bold'
    ).encode(
        y='y:Q',
        text='label:N'
    )

    final_chart = compute_area + memory_area + compute_lines + memory_lines +\
            zero_line + axis_labels

    if gen_boundaries:
      final_chart += gen_fusion_group_boundaries(df) 
      
    # === Combine everything
    return final_chart.properties(
        width=600,
        height=300,
        title=title
    )


  
def gen_campaign_block_throttle(equal_df, einsum_order, color_groups="fusion_group",
                       title="Campaign Diagram", color_scale=None, groups_color_scale=None,
                       gen_boundaries=True, debug=None):
    
    shared_tooltip = ['Einsum', 'runtime', 'og_runtime', 'Memory_Utilization', \
                      'Compute_Utilization', 'Compute_Utilization_plot', 'fusion_group']

    if debug != None:
      debug.print(f"Band colors are {equal_df[color_groups]}")
      debug.print("gen_campaign_block")
    
    base = alt.Chart(equal_df).mark_line().encode(
        x=alt.X('Time_Stamp:Q', title="Time (ms)"),
        y=alt.Y('Compute_Utilization_plot:Q', title="Compute Utilization")  # <-- fixed here
    ).transform_calculate(
        compute_limit_top="datum.Compute_Utilization_plot+0.25*datum.memory_well",
        compute_limit_bot="datum.Compute_Utilization_plot-0.25*datum.memory_well",
        y1_temp="datum.Memory_Utilization/4",
        y1="datum.y1_temp + datum.Compute_Utilization_plot",
        y2="-datum.y1_temp + datum.Compute_Utilization_plot"
    )

    tick = base.mark_rule(size=4).encode(
        x2="End_Time:Q",
        y=alt.Y('Compute_Utilization_plot:Q', title="Compute Utilization"
                 ),  # <-- fixed here
        color=alt.Color('Einsum:N', sort=einsum_order, scale=color_scale, title="Einsum"),
        tooltip=shared_tooltip
    )

    mem_area = base.mark_rect(opacity=0.4).encode(
        x2="End_Time:Q",
        y=alt.Y('y1:Q', title="Compute Utilization"),  # <-- added for consistency
        y2=alt.Y2('y2:Q'),
        tooltip=shared_tooltip,
        color=alt.Color('Einsum:N', sort=einsum_order, scale=color_scale, title="Einsum")
    )

    guard_min = base.mark_rule(color='black', strokeDash=[1, 1], size=2, interpolate='step-after').encode(
        x2="End_Time:Q",
        y=alt.Y('compute_limit_top:Q', title="Compute Utilization"),  
        tooltip=shared_tooltip,
        #color=alt.Color(f"{color_groups}:N", scale=groups_color_scale, title="Fusion Groups")
        color=alt.Color('Einsum:N', sort=einsum_order, scale=color_scale, title="Einsum")

    )

    guard_max = base.mark_rule(color='black', size=2, strokeDash=[1, 1], interpolate='step-after').encode(
        x2="End_Time:Q",
        y=alt.Y('compute_limit_bot:Q', title="Compute Utilization"),
        tooltip=shared_tooltip,
        #color=alt.Color(f"{color_groups}:N", scale=groups_color_scale, title="Fusion Groups")
        color=alt.Color('Einsum:N', sort=einsum_order, scale=color_scale, title="Einsum")

    )

    final_chart = guard_min + guard_max + mem_area + tick
    if gen_boundaries:
      final_chart += gen_fusion_group_boundaries(equal_df)
      

    return final_chart.properties(
        width=600,
        height=300,
        title=title
    )


def gen_campaign_block_throttle_log(equal_df, einsum_order, color_groups="fusion_group",
                                title="Campaign Diagram", color_scale=None, groups_color_scale=None,
                                   gen_boundaries=True, debug=None):
    
    shared_tooltip = ['Einsum', 'runtime', 'og_runtime', 'Memory_Utilization', 
                      'Compute_Utilization', 'Compute_Utilization_plot', 'fusion_group']

    if debug != None:
      debug.print(f"Band colors include groups: {equal_df[color_groups].unique()}")
      debug.print("gen_campaign_block")

    base = alt.Chart(equal_df).transform_calculate(
        Compute_Utilization_plot_shifted="datum.Compute_Utilization_plot + 1e-3",
        compute_limit_top="datum.Compute_Utilization_plot_shifted + 0.25 * datum.memory_well",
        compute_limit_bot="datum.Compute_Utilization_plot_shifted - 0.25 * datum.memory_well",
        y1_temp="datum.Memory_Utilization / 4 + 1e-3",
        y1="datum.y1_temp + datum.Compute_Utilization_plot_shifted",
        y2="-datum.y1_temp + datum.Compute_Utilization_plot_shifted"
    ).mark_line().encode(
        x=alt.X('Time_Stamp:Q', title="Time (ms)"),
        y=alt.Y('Compute_Utilization_plot_shifted:Q', 
                title="Compute Utilization", 
                scale=alt.Scale(type='log'))
    )

    tick = base.mark_rule(size=4).encode(
        x2="End_Time:Q",
        y=alt.Y('Compute_Utilization_plot_shifted:Q', 
                title="Compute Utilization", 
                scale=alt.Scale(type='log')),
        tooltip=shared_tooltip,
        color=alt.Color('Einsum:N', sort=einsum_order, scale=color_scale, title="Einsum")
    )

    mem_area = base.mark_rect(opacity=0.4).encode(
        x2="End_Time:Q",
        y=alt.Y('y1:Q', 
                title="Compute Utilization", 
                scale=alt.Scale(type='log')),
        y2=alt.Y2('y2:Q'),
        tooltip=shared_tooltip,
        color=alt.Color('Einsum:N', sort=einsum_order, scale=color_scale, title="Einsum")
    )

    guard_min = base.mark_rule(color='black', strokeDash=[1, 1], size=2, interpolate='step-after').encode(
        x2="End_Time:Q",
        y=alt.Y('compute_limit_top:Q', 
                title="Compute Utilization", 
                scale=alt.Scale(type='log')),
        tooltip=shared_tooltip,
        color=alt.Color('Einsum:N', sort=einsum_order, scale=color_scale, title="Einsum")
    )

    guard_max = base.mark_rule(color='black', size=2, strokeDash=[1, 1], interpolate='step-after').encode(
        x2="End_Time:Q",
        y=alt.Y('compute_limit_bot:Q', 
                title="Compute Utilization", 
                scale=alt.Scale(type='log')),
        tooltip=shared_tooltip,
        color=alt.Color('Einsum:N', sort=einsum_order, scale=color_scale, title="Einsum")
    )

    final_chart = guard_min + guard_max + mem_area + tick
    if gen_boundaries:
      final_chart += gen_fusion_group_boundaries(equal_df)

    

    return final_chart.properties(
        width=600,
        height=300,
        title=title
    )


# MP-style
def gen_campaign_block_dual_throttle(df, einsum_order, title="Shaded Utilization with Axis Labels", 
                                               color_scale=None, gen_boundaries=True):
    
    shared_tooltip = ['Einsum', 'runtime', 'og_runtime', 'Memory_Utilization', \
                      'Compute_Utilization', 'Compute_Utilization_plot', 'fusion_group']
    
    # === Compute shaded area (positive side)
    compute_area = alt.Chart(df).mark_rect(opacity=0.5).encode(
        x=alt.X('Starting_Time:Q', title="Time (ms)"),
        x2='End_Time:Q',
        y=alt.Y('cutil_start:Q', title="Memory Utilization | Compute Utilization", scale=alt.Scale(domain=[-1, 1])),
        y2='Compute_Utilization_plot:Q',
        color=alt.Color('Einsum:N', sort=einsum_order, scale=color_scale),
        tooltip=shared_tooltip
    )
    # .transform_calculate(
    #     zero='0'
    # )

    # === Memory shaded area (negative side)
    memory_area = alt.Chart(df).mark_rect(opacity=0.5).encode(
        x='Starting_Time:Q',
        x2='End_Time:Q',
        y='neg_mutil_start:Q',
        y2='neg_mem_util:Q',
        color=alt.Color('Einsum:N', sort=einsum_order, scale=color_scale),
        tooltip=shared_tooltip
    ).transform_calculate(
         zero='0',
         neg_mem_util='-datum.Memory_Utilization_plot',
         neg_mutil_start='-datum.mutil_start'

    )

    # === Compute and memory horizontal lines
    compute_lines = alt.Chart(df).mark_rule(size=2).encode(
        x='Starting_Time:Q',
        x2='End_Time:Q',
        y='Compute_Utilization_plot:Q',
        color=alt.Color('Einsum:N', sort=einsum_order, scale=color_scale),
        tooltip=shared_tooltip
    )

    memory_lines = alt.Chart(df).mark_rule(size=2).encode(
        x='Starting_Time:Q',
        x2='End_Time:Q',
        y='neg_mem_util:Q',
        color=alt.Color('Einsum:N', sort=einsum_order, scale=color_scale),
        tooltip=shared_tooltip
    ).transform_calculate(
        neg_mem_util='-datum.Memory_Utilization_plot'
    )

    # === Horizontal zero line
    zero_line = alt.Chart(pd.DataFrame({'y': [0]})).mark_rule(
        color='black', size=1
    ).encode(
        y='y:Q'
    )

    # === Proper label layer with per-row dy encoding
    label_df = pd.DataFrame({
        'y': [0.95, -0.95],
        'label': ['Compute Utilization', 'Memory Utilization']
    })

    axis_labels = alt.Chart(label_df).mark_text(
        fontSize=18,
        fontWeight='bold'
    ).encode(
        y='y:Q',
        text='label:N'
    )

    final_chart = compute_area + memory_area + compute_lines + memory_lines +\
            zero_line + axis_labels
  
    if gen_boundaries:
      final_chart += gen_fusion_group_boundaries(df)
      
    # === Combine everything
    return final_chart.properties(
        width=600,
        height=300,
        title=title
    )





def load_csv_as_cascade(csv_path, name="Imported Cascade", sequential=True):
    """
    Load a CSV file and convert it into a Cascade of Kernel instances.

    Parameters:
        csv_path (str): Path to CSV file
        name (str): Name of the resulting Cascade
        sequential (bool): Whether to assign start times automatically or read from CSV

    Returns:
        Cascade: a Cascade object composed of Kernels
    """
    df = pd.read_csv(csv_path)
    kernel_list = []

    for _, row in df.iterrows():
        kernel = Kernel(
            name=row["Einsum"],
            duration=float(row["runtime"]),
            compute_util=float(row["Compute_Utilization"]),
            bw_util=float(row["Memory_Utilization"]),
            fusion_group=row.get("fusion_group")  # Grab from CSV
        )

        fusion_group = row.get("fusion_group")

        fusion_type = row["fusion_type"] if "fusion_type" in df.columns else None
      
        # with CSV types 
        if not fusion_type:
          if fusion_group and fusion_group.strip() != "":
            fusion_type = FusionType.STRONG

        kernel.fusion_type = fusion_type
        
        wind_up = row["wind_up"] if "wind_up" in df.columns else 0.0
        kernel.windup = wind_up 

        if not sequential:
            kernel.start = float(row.get("Starting_Time", 0.0))

        kernel_list.append(kernel)
      
    return Cascade(kernels=kernel_list, name=name, sequential=sequential)

def load_df_as_cascade(df, name="Imported Cascade", sequential=True):
    """
    Convert a DataFrame into a Cascade of Kernel instances.

    Parameters:
        df (pd.DataFrame): DataFrame with kernel data
        name (str): Name of the resulting Cascade
        sequential (bool): Whether to assign start times automatically or use 'Starting_Time' from df

    Returns:
        Cascade: a Cascade object composed of Kernels
    """
    kernel_list = []

    for _, row in df.iterrows():
        einsum_name = str(row["Einsum"]).strip()
        fusion_group = str(row["fusion_group"]).strip() if "fusion_group" in df.columns else None

        duration = float(row["runtime"])
        compute_util = float(row["Compute_Utilization"])
        bw_util = float(row["Memory_Utilization"])
        wind_up = float(row["wind_up"]) if "wind_up" in df.columns else 0.0

        kernel = Kernel(
            name=einsum_name,
            duration=duration,
            compute_util=compute_util,
            bw_util=bw_util,
            fusion_group=fusion_group
        )

        fusion_type = row["fusion_type"] if "fusion_type" in df.columns else None
        if not fusion_type or (isinstance(fusion_type, str) and fusion_type.strip() == ""):
            if fusion_group and fusion_group != "":
                fusion_type = FusionType.STRONG
        kernel.fusion_type = fusion_type

        kernel.windup = wind_up

        if not sequential:
            kernel.start = float(row["Starting_Time"]) if "Starting_Time" in df.columns else 0.0

        kernel_list.append(kernel)

    return Cascade(kernels=kernel_list, name=name, sequential=sequential)




def stack_charts(chart_list):
  return alt.vconcat(*chart_list).resolve_scale(x='shared')


import copy

def _patch_x_axis_in_spec(spec: dict, *, hide_title=True, hide_ticks_labels=False) -> dict:
  """
  Recursively modify encoding.x.axis in a Vega-Lite spec.
  Does NOT add 'config', so it can be used inside vconcat/hconcat.
  """
  if not isinstance(spec, dict):
    return spec

  enc = spec.get("encoding")
  if isinstance(enc, dict) and isinstance(enc.get("x"), dict):
    x = enc["x"]
    axis = x.get("axis")
    if axis is None:
      axis = {}
      x["axis"] = axis
    if isinstance(axis, dict):
      if hide_title:
        axis["title"] = None
      if hide_ticks_labels:
        axis["labels"] = False
        axis["ticks"] = False
        axis["domain"] = False

  # Recurse into nested specs
  for key in ("layer", "hconcat", "vconcat", "concat"):
    if isinstance(spec.get(key), list):
      spec[key] = [_patch_x_axis_in_spec(s, hide_title=hide_title, hide_ticks_labels=hide_ticks_labels)
                  for s in spec[key]]

  if isinstance(spec.get("spec"), dict):
    spec["spec"] = _patch_x_axis_in_spec(spec["spec"], hide_title=hide_title, hide_ticks_labels=hide_ticks_labels)

  return spec

def hide_x_axis_without_config(chart, *, hide_ticks_labels=True):
  """
  Return a copy of `chart` with x-axis title removed (and optionally ticks/labels removed),
  implemented via encoding-axis edits (no config), so it can be concatenated.
  """
  spec = chart.to_dict()
  spec = _patch_x_axis_in_spec(copy.deepcopy(spec),
                              hide_title=True,
                              hide_ticks_labels=hide_ticks_labels)
  return alt.Chart.from_dict(spec)

def stack_charts_no_titles(chart_list, hide_ticks_labels=True):
  charts = list(chart_list)

  # Hide x-axis title (and optionally ticks/labels) for all but bottom chart
  for i in range(len(charts) - 1):
    charts[i] = hide_x_axis_without_config(charts[i], hide_ticks_labels=hide_ticks_labels)

  return alt.vconcat(*charts).resolve_scale(x="shared")
