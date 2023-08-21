import copy


class SimulationReport:
    def __init__(
        self,
        latency=None,
        theory_mac=None,
        theory_mem_access=None,
        compute_latency=None,
        memory_latency=None,
        cfg=None,
    ) -> None:
        """

        bound: either memory or computation
        """
        self.latency = latency
        self.theory_mac = theory_mac
        self.theory_mem_access = theory_mem_access
        self.compute_latency = compute_latency
        self.memory_latency = memory_latency
        # if memory_latency>compute_latency:
        #     self.bound='memory'
        # else:
        #     self.bound='computation'
        self.cfg = cfg
        self.info = ""

    def __add__(self, other):
        if other is None:
            return copy.deepcopy(self)
        new_report = SimulationReport()
        for att in [
            "latency",
            "compute_latency",
            "memory_latency",
            "theory_mac",
            "theory_mem_access",
        ]:
            if getattr(self, att) is not None and getattr(other, att) is not None:
                setattr(new_report, att, getattr(self, att) + getattr(other, att))
        return new_report

    def __str__(self):
        s = ""
        if self.latency is not None:
            s += f"latency={self.latency:.2e},"
        for att in [
            "compute_latency",
            "memory_latency",
            "theory_mac",
            "theory_mem_access",
        ]:
            if getattr(self, att) is not None:
                s += f"{att}={getattr(self,att):.2e},"
        s += self.info
        return s

    def print_cfg_C(self, **kwargs):
        for name, v in self.cfg.items():
            print(f"#define {name.upper()} {v}")
        for name, v in kwargs.items():
            print(f"#define {name.upper()} {v}")
