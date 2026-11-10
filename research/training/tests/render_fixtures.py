"""Native-session metadata required by simulated renderers and their recovery logs."""


class LifecycleReceipts:
    def _append(self, row):
        row["execution_policy"] = dict(self.execution_policy)
        super()._append(row)
