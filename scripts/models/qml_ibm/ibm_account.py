"""Save and check the IBM Quantum account used by the hardware QSVM runs.

save   Asks for the API key (hidden, never echoed or written to the shell history) and the
       instance CRN, and stores them with QiskitRuntimeService.save_account in
       ~/.qiskit/qiskit-ibm.json under --name (default account unless --no-default).
check  Logs in with the saved account and lists the operational backends (and the queue of
       --backend). It reads account and backend metadata only: no circuit is sent and no
       QPU time is used.

    python -m scripts.models.qml_ibm.ibm_account save --name zno
    python -m scripts.models.qml_ibm.ibm_account check --name zno --backend ibm_marrakesh

The key never goes into a YAML file, a notebook or the repository.
"""

from __future__ import annotations

import argparse
import sys
from getpass import getpass

CHANNEL = "ibm_quantum_platform"


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Save or check the IBM Quantum account (no circuits are run).")
    commands = parser.add_subparsers(dest="command", required=True)
    save = commands.add_parser("save", help="Store the API key and instance CRN (prompted).")
    save.add_argument("--name", default="zno", help="Name of the saved account (default zno).")
    save.add_argument("--overwrite", action="store_true", help="Replace a saved account with the same name.")
    save.add_argument("--no-default", action="store_true", help="Do not make it the default account.")
    check = commands.add_parser("check", help="Log in and list the backends (metadata only).")
    check.add_argument("--name", default=None, help="Saved account name (default: the default account).")
    check.add_argument("--backend", default=None, help="Backend whose status and queue to show.")
    check.add_argument("--force-ipv4", action="store_true")
    return parser.parse_args(argv)


def save_account(name: str, overwrite: bool, default: bool, token: str | None = None,
                 instance: str | None = None) -> None:
    from qiskit_ibm_runtime import QiskitRuntimeService

    token = token if token is not None else getpass("IBM Quantum API key (hidden): ").strip()
    instance = instance if instance is not None else input("Instance CRN (crn:v1:...): ").strip()
    if not token:
        raise SystemExit("[ibm-account] empty API key; nothing saved.")
    if not instance.startswith("crn:"):
        raise SystemExit("[ibm-account] the instance must be the CRN shown on the Instances page "
                         "(it starts with 'crn:'); nothing saved.")
    QiskitRuntimeService.save_account(channel=CHANNEL, token=token, instance=instance, name=name,
                                      set_as_default=default, overwrite=overwrite)
    print(f"[ibm-account] saved account '{name}'" + (" (default)" if default else "")
          + " in ~/.qiskit/qiskit-ibm.json. Next: python -m scripts.models.qml_ibm.ibm_account check "
          f"--name {name}")


def check_account(name: str | None, backend: str | None, force_ipv4: bool = False) -> None:
    from scripts.models.qml_ibm import qsvm_ibm_hardware as hw

    hw.configure_runtime_network(force_ipv4)
    service = hw.create_runtime_service(name, None)
    names = sorted(b.name for b in service.backends(operational=True, simulator=False))
    print(f"[ibm-account] login ok ({'account ' + name if name else 'default account'}); "
          f"operational backends: {', '.join(names) if names else 'none'}")
    if backend:
        if backend not in names:
            raise SystemExit(f"[ibm-account] {backend} is not among the operational backends of this account.")
        status = service.backend(backend).status()
        print(f"[ibm-account] {backend}: {status.status_msg}, {status.pending_jobs} jobs in the queue")
    print("[ibm-account] no circuit was sent; no QPU time was used.")


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "save":
            save_account(args.name, args.overwrite, not args.no_default)
        else:
            check_account(args.name, args.backend, args.force_ipv4)
    except ImportError as exc:
        raise SystemExit(f"[ibm-account] needs qiskit-ibm-runtime ({exc}); use the modeling environment.") from exc
    return 0


if __name__ == "__main__":
    sys.exit(main())
