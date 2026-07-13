
import copy

from typing import Tuple
from typing import List

# from deprecated import deprecated

from ruamel.yaml import YAML

import logging

from campaign_diagram.kernel_color import *
from campaign_diagram.kernel import *
from campaign_diagram.intervals import *
from campaign_diagram.fusion import FusionType
from fractions import Fraction


import pandas as pd


kernel_color_map = KernelColor()

class Cascade:
    """A class to manage a collection of Kernel instances."""

    # TODO: add support for  "+"
    # TODO: add support for len()

    def __init__(self, kernels: List[Kernel], name="", sequential=False):

        self.logger = logging.getLogger('campaign_diagram.cascade')
        self.logger.setLevel(logging.WARNING)

        self.name = name
        #print(f"Cascade.py - DEBUG: Loading data frame, sequential is {sequential}")

        # print(f"Before we assign starts...")
        # for kernel in kernels:
        #     print(f"{kernel.name},{kernel.start}, {kernel.end}\n")
          
        if sequential:
            self.assign_starts(kernels)
          
        # print(f"\n\After we assign starts...")
        # for kernel in kernels:
        #     print(f"{kernel.name},{kernel.start}, {kernel.end}")
          
        self.intervals = Intervals(kernels)
        self.logger.debug(f"Our cascade has {len(self.intervals.kernels)} kernels")


    @classmethod
    def fromYAML(cls, yaml_file):
        """ Creat a cascade from a YAML file """

        yaml = YAML()  # Initialize ruamel.yaml parser
        with open(yaml_file, 'r') as file:
            data = yaml.load(file)
            print(f"**************************************")
            print(f"Processing {yaml_file}")
            print(f"**************************************")


        cascade_data = data.get('cascade', {})
        name = cascade_data.get('name', 'Unnamed Cascade')

        kernels = []
        for kernel_data in cascade_data.get('kernels', []):
            kernel = Kernel(
                name=kernel_data.get('name'),
                duration=kernel_data.get('duration'),
                compute_util=kernel_data.get('compute_util'),
                bw_util=kernel_data.get('bw_util')
            )
            kernels.append(kernel)

        return cls(name=name,
                   kernels=kernels,
                   sequential=True)

    @classmethod
    def fromCSV(cls, csv_path, name="Imported Cascade", sequential=True):
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

        print(f"**************************************")
        print(f"Processing {csv_path}")
        print(f"**************************************")
      
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
          
        return cls(kernels=kernel_list, name=name, sequential=sequential)
  

    @classmethod
    def fromIntervals(cls, name, intervals):
        """ Create a csacade from an interval data structure """

        c = cls(name=name, kernels=[])
        c.intervals = intervals

        return c

    @property
    def kernels(self):
        """ Flatten intevals into a flat list of kernels """

        return self.intervals.flatten()

    def __len__(self):

        # TODO: Optimize?

        return len(self.kernels)


    def __iter__(self):
        """Return an iterator over the Kernel instances."""

        return iter(self.kernels)

    def duration(self):
       """ Find duration of cascade """

       return self.intervals.duration()

    def avg_compute_util(self):
        """ Return average compute util """

        return self.intervals.avg_compute_util()

    def avg_bw_util(self):
        """ Return average  bw util """

        return self.intervals.avg_bw_util()

    def is_sequential(self):

        last_end = 0

        for kernel in self.kernels:
            #print(f"    {kernel.name}: {kernel.start}, {kernel.end}")
            if kernel.start != last_end:
                #print(f"           FAILED HERE!")
                return False

            last_end = kernel.end

        return True

#    def sort(self):
#        """ Sort the kernels by start time for display """
#
#        self.kernels = sorted(self.kernels,
#                              key=lambda k: (k.start, -k.bw_util, k.compute_util, k.name))

    def assign_starts(self, kernels, offset=0):
        """ Assume the tasks are sequential and assign start times """

        last_end = offset
        for kernel in kernels:
            kernel.set_start(last_end)
            last_end = kernel.end
            # print(f"ASSIGN_STARTS: {kernel.name}: {kernel.start} -- {kernel.end}")

    def assign_colors(self):

        for kernel in self.kernels:
            name = kernel.name
            kernel.set_color(kernel_color_map.getColor(name))

    #@deprecated(reason="Cascade.split() has been replaced by Cascade.tile()")
    def split(self, parts):
        return self.tile(parts)

    def tile(self, parts):
        """Tile by splitting each task of a cascade into "parts" parts

        Note: This only works for a cascade that is a simple
        sequential series of kernels.

        Todo: Check invariant!

        """

        tile_duration_proportion = 1.0/parts #Fraction(1, parts)

        split_kernels = []

        for kernel in self.kernels:
            split_kernels.append(kernel.copy().scale_duration(tile_duration_proportion))

        split_cascade = Cascade(name=f"{self.name} (Tiled)",
                                kernels=[kernel.clone() for kernel in parts*split_kernels],
                                sequential=True)

        return split_cascade


    def tile_fusion_group(self, fusion_group, parts):
        """
        Tile only the specified fusion group into `parts` segments.
        All other kernels remain unchanged. Start times are reassigned 
        globally to preserve sequential execution.
        """
        tiled_kernels = []

        tile_duration_proportion = 1.0/parts #Fraction(1, parts)


        split_kernels = []

        for kernel in self.kernels:
            if kernel.fusion_group == fusion_group:
                for _ in range(parts):
                    new_kernel = kernel.copy().scale_duration(tile_duration_proportion) 
                    tiled_kernels.append(new_kernel)
            else:
                tiled_kernels.append(kernel.copy())

        split_cascade = Cascade(name=f"{self.name} (Tiled)",
                                kernels=[kernel.clone() for kernel in parts*tiled_kernels],
                                sequential=True)

        return split_cascade

    
    
        # # Assign new start times sequentially
        # last_end = 0
        # for kernel in tiled_kernels:
        #     kernel.set_start(last_end)
        #     last_end = kernel.end
    
        # return Cascade(
        #     name=f"{self.name} (FG {fusion_group} tiled x{parts})",
        #     kernels=tiled_kernels,
        #     sequential=False
        # )

    def tile_fusion_group_with_dependencies(self, fusion_group, dependency_structure, parts, start_time=0):
        """
        Tile the specified fusion group using a sequential dependency structure.
        Other fusion groups are preserved. Tiles are executed serially and sequentially.
    
        Parameters:
        - fusion_group: fusion group ID (str or int)
        - dependency_structure: list of list of Einsum names in sequential execution order
        - parts: number of tiles
        - start_time: time to start tiling from
    
        Returns:
        - new Cascade with this fusion group tiled in place
        - end_time after this group's tiled execution finishes
        """
        # Separate kernels
        target_kernels = [k for k in self.kernels if k.fusion_group == fusion_group]
        other_kernels = [k.copy() for k in self.kernels if k.fusion_group != fusion_group]
    
        einsum_map = {k.name: k for k in target_kernels}
    
        # Sanity check
        for stage in dependency_structure:
            for einsum_name in stage:
                if einsum_name not in einsum_map:
                    raise ValueError(f"Einsum '{einsum_name}' not found in fusion group '{fusion_group}'")
    
        new_kernels = []
    
        time = start_time
        for tile_idx in range(parts):
            for stage in dependency_structure:
                for einsum_name in stage:
                    base_kernel = einsum_map[einsum_name]
                    k = base_kernel.copy().scale_duration( 1.0/parts) #Fraction(1, parts))
                    k.set_start(time)
                    time = k.end
                    new_kernels.append(k)
    
        # Merge and sort
        final_kernels = self._shift_fusion_groups(other_kernels + new_kernels, fusion_group, shift_amount=(time - start_time))

        return (
            Cascade(
                name=f"{self.name} (FG {fusion_group} DepTiled x{parts})",
                kernels=final_kernels,
                sequential=False
            ),
            time  # New end time
        )



    def pipeline(self, stages=2, spread=False, create_spacers=True):
        """Pipeline a set of tasks

        Note: Input must be a sequential set of tiles
        Note: This is not meaningful after a cascade is throttled

        """

        assert self.is_sequential()

        tasks = []

        orig_kernels = copy.deepcopy(self.kernels)

        for stage in range(stages):
            name = orig_kernels[stage].name
            if create_spacers:
              task = self._create_spacers(stage, name)
              task.extend(orig_kernels[stage::stages])
              task.extend(self._create_spacers(stages-stage-1, name))
            else:
              task = orig_kernels[stage::stages]
            tasks.append(task)

        # Start with a default previous_end value of zero
        new_kernels = []
        previous_end = 0

        # Iterate over both lists in tandem
        for kernels in zip(*tasks):

            max_duration = max([kernel.duration for kernel in kernels])

            for kernel in kernels:

                # Just skip kernels with duration == 0
                if kernel.duration == 0:
                    continue

                kernel.set_start(previous_end)

                if spread:
                    if kernel.duration !=0 and kernel.duration < max_duration:
                        kernel.dilate(max_duration/kernel.duration)

                new_kernels.append(kernel)

            # TODO: Handle resource overutilization

            previous_end += max_duration

        t = Cascade(name=f"{self.name} (Pipelined)",
                    kernels=new_kernels)

        return t

    def untiled_pipeline(self, stages=2, spread=False, create_spacers=False):
        """Pipeline a set of tasks

        Note: Input must be a sequential set of tiles
        Note: This is not meaningful after a cascade is throttled

        """

        assert self.is_sequential(), f"Failed: {self.name}"

        tasks = []

        orig_kernels = copy.deepcopy(self.kernels)
        # print(f"Original kernels: {orig_kernels}")
        # print(f"Stages is: {stages}")

        # stages is the number of Einsums being grouped in a pipeline
        # originally, this logic was trying to find every (stages) # of einsums to group within the same "task"
        for stage in range(stages):
            name = orig_kernels[stage].name
            # Spacers don't let the code find the write Einsums to group.
            # if create_spacers:
            #   task = self._create_spacers(stage, name)
            #   task.extend(orig_kernels[stage::stages])
            #   task.extend(self._create_spacers(stages-stage-1, name))
            # else:
            task = orig_kernels[stage::stages]
            tasks.append(task)

        # Start with a default previous_end value of zero
        new_kernels = []
        previous_end = 0
      
        #print(tasks)

      # Iterate over both lists in tandem
        for kernels in zip(*tasks):

            # Figure out how long this will take
            max_duration = max([kernel.duration for kernel in kernels])
          
            # Within a group, we need to do this...
            previous_windup = 0

            for kernel in kernels:

                # Just skip kernels with duration == 0
                if kernel.duration == 0:
                    continue

                kernel.set_start(previous_end)

                if spread:
                    #print(f"We were in spread for {kernel.name}, {kernel.duration}, {max_duration}")
                    if kernel.duration !=0 and kernel.duration < max_duration:
                        kernel.dilate(max_duration/kernel.duration)

                new_kernels.append(kernel)
                # get the windup of this kernel 
                #  so the next kernel knows when to start
                previous_windup = kernel.windup
                # the previous fusion group end + this kernel's windup
                #   is when the next kernel should start
                previous_end += previous_windup

            # TODO: Handle resource overutilization -- done with throttle call

            # We have group -1, 0 (this group), and 1 (next group)
            # old previous_end is when -1 ended.
            # this group has a max duration that we add to the previous end
            # BUT we also had windups included
            previous_end += max_duration

        t = Cascade(name=f"{self.name} (Pipelined)",
                    kernels=new_kernels)
        #print(f"Returning this new kernel {t}")

        return t


    def pipeline_fusion_group(self, fusion_group: str, stages: int = 2, spread: bool = False):
        """
        Pipeline all kernels that belong to `fusion_group` using the same
        algorithm as `self.pipeline`, but leave the rest of the cascade untouched.
    
        Returns a NEW Cascade.
        """
        # 1. Split kernels into [before] [target group] [after]
        before, group, after = [], [], []
        seen = False

        # loop through each kernel
        for k in self.kernels:
            # if this kernel is in our fusion group, add it to our
            # group to process
            if k.fusion_group == fusion_group:
                group.append(copy.deepcopy(k))
                seen = True
            elif not seen:
                before.append(copy.deepcopy(k))
            else:
                after.append(copy.deepcopy(k))
    
        # Nothing to do if the fusion group wasn’t found
        if not group:
            return self

        # We need to do the start and end BEFORE we pass it to pipeline. Pipeline does not preserve times
        # old_group_start = group[0].start
        # old_group_end   = group[-1].end
        min_start_idx = min(enumerate(group), key=lambda x: x[1].start)[0]
        max_end_idx   = max(enumerate(group), key=lambda x: x[1].end)[0]
        
        old_group_start = group[min_start_idx].start
        old_group_end   = group[max_end_idx].end

        # 2. Pipeline the target group (reuse existing pipeline logic)
        group_cascade   = Cascade(group, name=f"{fusion_group}_orig", sequential=True)
        piped_group     = group_cascade.pipeline(stages=stages, spread=spread, create_spacers=True).throttle()
        # print(f"We just pipelined {fusion_group}_orig!")
        # print(f"We have before: {before}, group: {group}, after: {after}")
        # print(f"Now we have {piped_group}")
      
        # 3. Align pipelined group to ORIGINAL start time
        shift_offset    = old_group_start - min(k.start for k in piped_group.kernels)
        for k in piped_group.kernels:
            k.start += shift_offset
    
        # 4. Shift every kernel after the group to keep the timeline contiguous
        new_group_end   = max(k.end for k in piped_group.kernels)
        delta           = new_group_end - old_group_end          # ≥ 0
    
        for k in after:
            k.start += delta
    
        # 5. Re-assemble and return a new cascade
        new_kernels = before + piped_group.kernels + after
        return Cascade(new_kernels, name=f"{self.name} (FG {fusion_group} pipelined)")
  
    def compact_starts(self):
        """Make the cascade sequential-contiguous in-place."""
        prev_end = 0.0
        for k in sorted(self.kernels, key=lambda x: (x.start, x.name)):
            k.start = prev_end
            prev_end = k.end
    
    #####################################################################
    # Alternate Approach:
    # - Instead of tiling, pass in a "wind-up" time for each stage
    # - Indicate the dependency of the Einsums in the fusion group
    # - If two (or more Einsums) are dependent, start the first one
    # - then start the next one after the wind up time
    # - we can apply throttling later to add dashed lines 

    def untiled_pipeline_group(self, fusion_group: str,
                         stages: int = 2, 
                         spread: bool = False) -> "Cascade":
        """
        Pipeline all kernels that belong to `fusion_group` using the same
        algorithm as `self.pipeline`, but leave the rest of the cascade untouched.

        Do NOT use tiles. 
         
        Returns a NEW Cascade.
        """      

        # 1. Split kernels into [before] [target group] [after]
        before, group, after = [], [], []
        seen = False

        # loop through each kernel
        for k in self.kernels:
            # if this kernel is in our fusion group, add it to our
            # group to process
            if k.fusion_group == fusion_group:
                group.append(copy.deepcopy(k))
                seen = True
            elif not seen:
                before.append(copy.deepcopy(k))
            else:
                after.append(copy.deepcopy(k))

        # print(f"Processing {fusion_group}")
        # for kernel in group:
        #   print(kernel)

        # Nothing to do if the fusion group wasn’t found
        if not group:
            return self


      
        # We need to do the start and end BEFORE we pass it to pipeline. Pipeline does not preserve times
        # old_group_start = group[0].start
        # old_group_end   = group[-1].end
        min_start_idx = min(enumerate(group), key=lambda x: x[1].start)[0]
        max_end_idx   = max(enumerate(group), key=lambda x: x[1].end)[0]
        
        old_group_start = group[min_start_idx].start
        old_group_end   = group[max_end_idx].end

        # 2. Pipeline the target group (reuse existing pipeline logic)
        # Note - I need to change the pipeline function here to use a wind-up time
        group_cascade   = Cascade(group, name=f"{fusion_group}_orig", sequential=True)
        piped_group     = group_cascade.untiled_pipeline(stages=stages,\
                                                         spread=spread, create_spacers=True)
        piped_group = piped_group.throttle()
        #print("\n\n\n")
      
        # 3. Align pipelined group to ORIGINAL start time
        shift_offset    = old_group_start - min(k.start for k in piped_group.kernels)

        for k in piped_group.kernels:
            k.start += shift_offset
    
        # 4. Shift every kernel after the group to keep the timeline contiguous
        new_group_end   = max(k.end for k in piped_group.kernels)
        delta           = new_group_end - old_group_end          # ≥ 0
    
        for k in after:
            k.start += delta
    
        # 5. Re-assemble and return a new cascade
        new_kernels = before + piped_group.kernels + after
        #print(f"We are creating a cascade from {new_kernels}")
        return Cascade(new_kernels, name=f"{self.name} (FG {fusion_group} pipelined)")
        

    def set_kernel_windup(self, kernel_name, windup):
      for k in self.kernels:
        if k.name == kernel_name:
          k.set_windup(windup)
      
    # def pipeline_fusion_group(self, fusion_group: str, *args, **kwargs):
    #     """
    #     Pipeline all kernels in `fusion_group`, then push everything
    #     that starts after the group so that time is contiguous.
    #     Returns a new Cascade.
    #     """
    #     # 1. isolate kernels belonging to this fusion group
    #     fg_kernels   = [k for k in self.kernels if k.fusion_group == fusion_group]
    #     non_fg       = [k for k in self.kernels if k.fusion_group != fusion_group]
    
    #     # 2. pipeline the fusion-group kernels (use your existing routine)
    #     pipelined_fg = Cascade(fg_kernels, name=f"{fusion_group}-piped", sequential=True)\
    #                       .pipeline_fusion_group_helper(*args, **kwargs)  # stages, spread, …
    
    #     # 3. compute the time delta between old end and new end
    #     old_end = max(k.end for k in fg_kernels)
    #     new_end = max(k.end for k in pipelined_fg.kernels)
    #     delta   = new_end - old_end                      # >0 when pipelined group grew
    
    #     # 4. shift every non-fg kernel that starts ≥ old_end
    #     shifted_non_fg = []
    #     for k in non_fg:
    #         if k.start >= old_end:
    #             k = k.copy()
    #             k.start += delta
    #         shifted_non_fg.append(k)
    
    #     # 5. combine and return a new cascade
    #     return Cascade(pipelined_fg.kernels + shifted_non_fg,
    #                    name=f"{self.name} (fg={fusion_group} pipelined)")


    def throttle_fusion_group(self, fusion_group):
        """
        Throttle just the specified fusion group using Intervals.throttle().
        """
        from copy import deepcopy
        from campaign_diagram.intervals import Intervals
    
        # Isolate the group
        group_kernels = [k for k in self.kernels if k.fusion_group == fusion_group]
        other_kernels = [k.copy() for k in self.kernels if k.fusion_group != fusion_group]
    
        group_intervals = Intervals(group_kernels)
        group_intervals.throttle()
        throttled_kernels = group_intervals.flatten()

        group_end_before = max(k.end for k in group_kernels)
        group_end_after  = max(k.end for k in throttled_kernels)
        shifted_kernels = self._shift_fusion_groups(other_kernels + throttled_kernels, fusion_group, shift_amount=(group_end_after - group_end_before))


        return Cascade(
            name=f"{self.name} (FG {fusion_group} Throttled)",
            kernels=shifted_kernels,
            sequential=False
        )

  
    def _create_spacers(self, count, name=None):

        spacer = Kernel(name=name,
                        duration=0,
                        compute_util=0,
                        bw_util=0)

        spacers =  [copy.copy(spacer) for _ in range(count)]

        return spacers

    def _shift_fusion_groups(self, base_kernels, modified_group, shift_amount):
        """
        Shift start times of all kernels in fusion groups that start after `modified_group`
        by `shift_amount`.
        """
        modified_group_end = max(k.end for k in base_kernels if k.fusion_group == modified_group)
    
        new_kernels = []
        for k in base_kernels:
            if k.fusion_group == modified_group:
                new_kernels.append(k)
            elif k.start >= modified_group_end:
                k = k.copy()
                k.set_start(k.start + shift_amount)
                new_kernels.append(k)
            else:
                new_kernels.append(k.copy())
    
        return sorted(new_kernels, key=lambda k: (k.start, -k.bw_util, k.compute_util, k.name))

  
    def throttle(self):
        """Throttle a cascde to keep within resource constraints."""

        self.logger.debug("Starting throttle")

        new_intervals = copy.deepcopy(self.intervals)

        # Throttle the kernels in place
        new_intervals.throttle()

        ########################################################################
        # Flag every kernel in the throttled cascade as was_transformed=True.  #
        # The renderer's `show_ideal_dashes` kwarg uses this to decide whether #
        # to keep the dashed extensions for a given kernel. Throttle is a     #
        # user-facing transform: the user explicitly opted in by calling      #
        # .throttle(), so every resulting kernel is "transformed" -- even     #
        # those whose duration didn't materially change (e.g. kernels that    #
        # weren't oversubscribed before the call). This matches the user's    #
        # mental model: "if throttle/dilate was called, show the dashes".     #
        ########################################################################
        for kernel in new_intervals.flatten():
            kernel.was_transformed = True

        return Cascade.fromIntervals(name=f"{self.name} (Throttled)",
                                     intervals=new_intervals)

    def set_fusion_type():
      """
      For each kernel in the cascade, determine what type of fusion it has
      with the next Einsum.

      Note that this only works for Einsums that are fused with the next Einsum.
      It does not work for multi-hop-Einsum fusion (where an Einsum has downstream 
      Einsums that are not directly after itself.)


      TODO: make this work for any fusion scenario. 
      """
      pass
      # for i, kernel in enumerate(self.kernels):
      #   if kernel.fusion_type != None:
          
          
  

    def pretty_print(self, intervals=False):

        print(f"Cascade: {self.name}")

        if not intervals:
            for kernel in self.kernels:
                print(f"{kernel}\n")
        else:
            self.intervals.pretty_print()

    def __str__(self):
        """Returns a human-readable string representation of the CampaignDiagkernelram's state."""

        kernel_states = "\n".join([str(kernel) for kernel in self.kernels])
        return f"Cascade: {self.name} with kernels:\n{kernel_states}"

    def to_df(self):
        return pd.DataFrame([kernel.to_dict() for kernel in self.kernels])
      