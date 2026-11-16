"""ibm_account: the key is prompted and passed to save_account; check reads metadata only."""

import types
import unittest
from unittest import mock

from scripts.models.qml_ibm import ibm_account
from scripts.models.qml_ibm import qsvm_ibm_hardware as hw


class FakeRuntime:
    saved: list = []

    @classmethod
    def save_account(cls, **kwargs):
        cls.saved.append(kwargs)


class IbmAccountTests(unittest.TestCase):
    def setUp(self):
        FakeRuntime.saved = []
        module = types.SimpleNamespace(QiskitRuntimeService=FakeRuntime)
        patcher = mock.patch.dict("sys.modules", {"qiskit_ibm_runtime": module})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_save_prompts_for_the_key_and_the_crn(self):
        with mock.patch.object(ibm_account, "getpass", return_value=" secret-key \n") as hidden, \
                mock.patch("builtins.input", return_value="crn:v1:bluemix:public:quantum-computing:x::"), \
                mock.patch("builtins.print") as printed:
            ibm_account.main(["save", "--name", "zno"])
        hidden.assert_called_once()
        self.assertEqual(FakeRuntime.saved, [{
            "channel": "ibm_quantum_platform", "token": "secret-key",
            "instance": "crn:v1:bluemix:public:quantum-computing:x::", "name": "zno",
            "set_as_default": True, "overwrite": False}])
        self.assertNotIn("secret-key", " ".join(str(call) for call in printed.call_args_list))

    def test_save_refuses_an_instance_that_is_not_a_crn(self):
        with mock.patch.object(ibm_account, "getpass", return_value="key"), \
                mock.patch("builtins.input", return_value="my-instance"):
            with self.assertRaisesRegex(SystemExit, "CRN"):
                ibm_account.main(["save"])
        self.assertEqual(FakeRuntime.saved, [])

    def test_check_lists_backends_without_running_circuits(self):
        backend = types.SimpleNamespace(name="ibm_marrakesh")
        status = types.SimpleNamespace(status_msg="active", pending_jobs=3)
        service = mock.Mock()
        service.backends.return_value = [backend, types.SimpleNamespace(name="ibm_fez")]
        service.backend.return_value.status.return_value = status
        with mock.patch.object(hw, "create_runtime_service", return_value=service) as create, \
                mock.patch("builtins.print") as printed:
            ibm_account.main(["check", "--name", "zno", "--backend", "ibm_marrakesh"])
        create.assert_called_once_with("zno", None)
        text = " ".join(str(call) for call in printed.call_args_list)
        self.assertIn("ibm_fez, ibm_marrakesh", text)
        self.assertIn("3 jobs in the queue", text)
        names = [call[0] for call in service.mock_calls]
        self.assertEqual(names, ["backends", "backend", "backend().status"])  # metadata only, no run/job

    def test_check_refuses_an_unknown_backend(self):
        service = mock.Mock()
        service.backends.return_value = [types.SimpleNamespace(name="ibm_fez")]
        with mock.patch.object(hw, "create_runtime_service", return_value=service), \
                mock.patch("builtins.print"):
            with self.assertRaisesRegex(SystemExit, "ibm_marrakesh"):
                ibm_account.main(["check", "--backend", "ibm_marrakesh"])


if __name__ == "__main__":
    unittest.main()
