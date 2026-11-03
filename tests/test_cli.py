"""Tests for cli.py — command-line interface."""
import pytest

from cli import _build_parser


class TestCLIArgParsing:
    def test_empty_args_defaults(self):
        parser = _build_parser()
        args = parser.parse_args([])
        # GWAS required args are validated in main(), not parse_args
        assert args.vcf is None

    def test_required_args_present(self):
        parser = _build_parser()
        args = parser.parse_args([
            "--vcf", "test.vcf",
            "--pheno", "pheno.csv",
            "--trait", "Yield",
            "--output", "results/",
        ])
        assert args.vcf == "test.vcf"
        assert args.pheno == "pheno.csv"
        assert args.trait == "Yield"
        assert args.output == "results/"

    def test_default_values(self):
        parser = _build_parser()
        args = parser.parse_args([
            "--vcf", "x.vcf", "--pheno", "p.csv", "--trait", "Y", "--output", "o/",
        ])
        assert args.maf == 0.05
        assert args.miss == 0.10
        assert args.mac == 5
        assert args.ind_miss == 0.20
        assert args.info_thresh == 0.0
        assert args.n_pcs == 0  # auto-PC selector removed; fixed default is 0 PCs
        assert args.model == ["mlm"]
        assert args.no_report is False
        assert args.no_plots is False

    def test_model_choices(self):
        parser = _build_parser()
        args = parser.parse_args([
            "--vcf", "x.vcf", "--pheno", "p.csv", "--trait", "Y", "--output", "o/",
            "--model", "mlm", "farmcpu",
        ])
        assert "mlm" in args.model
        assert "farmcpu" in args.model

    def test_invalid_model_rejected(self):
        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([
                "--vcf", "x.vcf", "--pheno", "p.csv", "--trait", "Y", "--output", "o/",
                "--model", "invalid_model",
            ])

    def test_farmcpu_final_scan_flag_removed(self):
        # D-103: the FarmCPU final scan is frozen to the published OLS scan; the
        # --farmcpu-final-scan selector was removed, so argparse must reject it.
        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([
                "--vcf", "x.vcf", "--pheno", "p.csv", "--trait", "Y", "--output", "o/",
                "--farmcpu-final-scan", "mlm",
            ])

    def test_qc_overrides(self):
        parser = _build_parser()
        args = parser.parse_args([
            "--vcf", "x.vcf", "--pheno", "p.csv", "--trait", "Y", "--output", "o/",
            "--maf", "0.01", "--miss", "0.20", "--mac", "3",
            "--info-thresh", "0.8",
        ])
        assert args.maf == 0.01
        assert args.miss == 0.20
        assert args.mac == 3
        assert args.info_thresh == 0.8

    def test_flags(self):
        parser = _build_parser()
        args = parser.parse_args([
            "--vcf", "x.vcf", "--pheno", "p.csv", "--trait", "Y", "--output", "o/",
            "--no-report", "--no-plots", "-v",
        ])
        assert args.no_report is True
        assert args.no_plots is True
        # P11: --drop-alt removed (ALT is always excluded); flag no longer exists.
        assert args.verbose is True

    def test_export_qc_flag(self):
        parser = _build_parser()
        args = parser.parse_args([
            "--vcf", "x.vcf", "--pheno", "p.csv", "--trait", "Y", "--output", "o/",
            "--export-qc",
        ])
        assert args.export_qc is True

    def test_export_qc_default_false(self):
        parser = _build_parser()
        args = parser.parse_args([
            "--vcf", "x.vcf", "--pheno", "p.csv", "--trait", "Y", "--output", "o/",
        ])
        assert args.export_qc is False

    def test_auto_pc_flags_removed(self):
        # The lambda-GC auto-PC selector was removed; its flags no longer
        # parse and the fixed default is 0 PCs (no auto-selection).
        parser = _build_parser()
        args = parser.parse_args([
            "--vcf", "x.vcf", "--pheno", "p.csv", "--trait", "Y", "--output", "o/",
        ])
        assert args.n_pcs == 0
        assert not hasattr(args, "auto_pcs")
        for _flag in ("--auto-pcs", "--pc-strategy", "--max-pcs",
                      "--pc-band-lo", "--pc-parsimony-tol"):
            with pytest.raises(SystemExit):
                parser.parse_args([
                    "--vcf", "x.vcf", "--pheno", "p.csv", "--trait", "Y",
                    "--output", "o/", _flag, "1",
                ])

    def test_subsampling_defaults(self):
        parser = _build_parser()
        args = parser.parse_args([
            "--vcf", "x.vcf", "--pheno", "p.csv", "--trait", "Y", "--output", "o/",
        ])
        assert args.subsampling is False
        assert args.boot_reps == 50
        assert args.boot_frac == 0.80
        assert args.boot_thresh == 1e-4
        assert args.boot_jobs == 1

    def test_subsampling_overrides(self):
        parser = _build_parser()
        args = parser.parse_args([
            "--vcf", "x.vcf", "--pheno", "p.csv", "--trait", "Y", "--output", "o/",
            "--subsampling", "--boot-reps", "50", "--boot-frac", "0.75",
            "--boot-thresh", "1e-3", "--boot-jobs", "-1",
        ])
        assert args.subsampling is True
        assert args.boot_reps == 50
        assert args.boot_frac == 0.75
        assert args.boot_thresh == 1e-3
        assert args.boot_jobs == -1

    def test_ld_defaults(self):
        parser = _build_parser()
        args = parser.parse_args([
            "--vcf", "x.vcf", "--pheno", "p.csv", "--trait", "Y", "--output", "o/",
        ])
        assert args.ld_r2 == 0.6
        assert args.ld_flank_kb is None
        assert args.ld_seed_p == 1e-5
        assert args.ld_top_n == 10
        assert args.hap_perms == 1000
        assert args.no_annotation is False

    def test_ld_overrides(self):
        parser = _build_parser()
        args = parser.parse_args([
            "--vcf", "x.vcf", "--pheno", "p.csv", "--trait", "Y", "--output", "o/",
            "--ld-r2", "0.4", "--ld-flank-kb", "200", "--ld-seed-p", "1e-3",
            "--ld-top-n", "5", "--hap-perms", "1000", "--no-annotation",
        ])
        assert args.ld_r2 == 0.4
        assert args.ld_flank_kb == 200
        assert args.ld_seed_p == 1e-3
        assert args.ld_top_n == 5
        assert args.hap_perms == 1000
        assert args.no_annotation is True

    def test_per_model_pcs(self):
        parser = _build_parser()
        args = parser.parse_args([
            "--vcf", "x.vcf", "--pheno", "p.csv", "--trait", "Y", "--output", "o/",
            "--n-pcs-mlm", "6", "--n-pcs-mlmm", "3", "--n-pcs-farmcpu", "8",
        ])
        assert args.n_pcs_mlm == 6
        assert args.n_pcs_mlmm == 3
        assert args.n_pcs_farmcpu == 8

    def test_interactive_flag(self):
        parser = _build_parser()
        args = parser.parse_args([
            "--vcf", "x.vcf", "--pheno", "p.csv", "--trait", "Y", "--output", "o/",
            "--interactive",
        ])
        assert args.interactive is True

    def test_interactive_default_off(self):
        parser = _build_parser()
        args = parser.parse_args([
            "--vcf", "x.vcf", "--pheno", "p.csv", "--trait", "Y", "--output", "o/",
        ])
        assert args.interactive is False

    def test_sig_thresh_default(self):
        parser = _build_parser()
        args = parser.parse_args([
            "--vcf", "x.vcf", "--pheno", "p.csv", "--trait", "Y", "--output", "o/",
        ])
        assert args.sig_thresh == "bonferroni"

    def test_sig_thresh_choices(self):
        parser = _build_parser()
        for choice in ("meff", "bonferroni", "fdr"):
            args = parser.parse_args([
                "--vcf", "x.vcf", "--pheno", "p.csv", "--trait", "Y", "--output", "o/",
                "--sig-thresh", choice,
            ])
            assert args.sig_thresh == choice

    def test_sig_thresh_invalid_rejected(self):
        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([
                "--vcf", "x.vcf", "--pheno", "p.csv", "--trait", "Y", "--output", "o/",
                "--sig-thresh", "invalid",
            ])

    def test_help_exits_cleanly(self):
        parser = _build_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["--help"])
        assert exc_info.value.code == 0


class TestInteractiveHelpers:
    def test_build_equivalent_command_defaults(self):
        from cli import _build_equivalent_command, _build_parser
        parser = _build_parser()
        args = parser.parse_args([])

        args.vcf = "data.vcf.gz"
        args.pheno = "pheno.csv"
        args.trait = "Yield"
        args.output = "results/"
        cmd = _build_equivalent_command(args)
        assert "python cli.py" in cmd
        assert "--vcf data.vcf.gz" in cmd
        assert "--trait Yield" in cmd
        # default model is mlm — should NOT appear
        assert "--model" not in cmd

    def test_build_equivalent_command_multi_model(self):
        from cli import _build_equivalent_command, _build_parser
        parser = _build_parser()
        args = parser.parse_args([])

        args.vcf = "data.vcf.gz"
        args.pheno = "pheno.csv"
        args.trait = "Yield"
        args.output = "results/"
        args.model = ["mlm", "mlmm", "farmcpu"]
        args.subsampling = True
        args.boot_reps = 50
        args.boot_jobs = -1
        cmd = _build_equivalent_command(args)
        assert "--model mlm mlmm farmcpu" in cmd
        assert "--subsampling" in cmd
        assert "--boot-reps 50" in cmd
        assert "--boot-jobs -1" in cmd

    def test_build_equivalent_command_sig_thresh(self):
        from cli import _build_equivalent_command, _build_parser
        parser = _build_parser()
        args = parser.parse_args([])

        args.vcf = "data.vcf.gz"
        args.pheno = "pheno.csv"
        args.trait = "Yield"
        args.output = "results/"
        # default bonferroni — should NOT appear
        args.sig_thresh = "bonferroni"
        cmd = _build_equivalent_command(args)
        assert "--sig-thresh" not in cmd
        # non-default — should appear
        args.sig_thresh = "meff"
        cmd = _build_equivalent_command(args)
        assert "--sig-thresh meff" in cmd

    def test_peek_phenotype_missing_file(self):
        from cli import _peek_phenotype
        result = _peek_phenotype("/nonexistent/file.csv")
        assert result is None

    def test_print_summary_no_error(self):
        """_print_summary should not raise for valid args."""
        from cli import _print_summary, _build_parser
        parser = _build_parser()
        args = parser.parse_args([])

        args.vcf = "test.vcf"
        args.pheno = "test.csv"
        args.trait = "Y"
        args.output = "out/"
        args.model = ["mlm"]
        # Should not raise
        _print_summary(args)


class TestOutputDirWriteCheck:
    """CC-1 (2-R1.1): an unwritable --output must fail fast, before any analysis.

    ``mkdir(exist_ok=True)`` on an existing folder never tests writability and the
    results ZIP is the only write into ``--output``, so an unwritable Docker mount
    used to fail with a PermissionError traceback only after the whole run.
    """

    @staticmethod
    def _args(tmp_path, output):
        vcf = tmp_path / "in.vcf"
        vcf.write_text("##fileformat=VCFv4.2\n")
        pheno = tmp_path / "p.csv"
        pheno.write_text("ID,Y\nS1,1.0\n")
        return _build_parser().parse_args([
            "--vcf", str(vcf), "--pheno", str(pheno), "--trait", "Y",
            "--output", str(output),
        ])

    @staticmethod
    def _no_analysis(monkeypatch):
        """Fail loudly if the run gets as far as reading the VCF."""
        import gwas.io

        def _boom(*a, **k):
            raise AssertionError("analysis started despite an unwritable --output")

        monkeypatch.setattr(gwas.io, "load_vcf_cached", _boom)

    def test_output_path_that_is_a_file_exits_before_analysis(self, tmp_path, monkeypatch):
        from cli import run_pipeline
        self._no_analysis(monkeypatch)
        blocker = tmp_path / "not_a_dir"
        blocker.write_text("x")
        with pytest.raises(SystemExit) as ei:
            run_pipeline(self._args(tmp_path, blocker))
        msg = str(ei.value.code)
        assert "cannot write to output directory" in msg
        assert '--user "$(id -u):$(id -g)"' in msg

    def test_denied_create_exits_after_one_attempt(self, tmp_path, monkeypatch):
        """Simulate the Docker case (folder exists, EACCES on create) portably.

        Exactly one create attempt: tempfile.mkstemp would retry a PermissionError up
        to TMP_MAX (2**31 - 1) names on Windows, which hangs instead of exiting. The
        stub refuses a fourth call, so a reintroduced retry loop fails fast here.
        """
        import os
        from cli import _ensure_writable_output_dir

        calls = []

        def _denied(path, flags, mode=0o777, *a, **k):
            calls.append(path)
            if len(calls) > 3:
                raise AssertionError("write probe retries the create instead of exiting")
            raise PermissionError(13, "Permission denied")

        out = tmp_path / "results"
        out.mkdir()
        monkeypatch.setattr(os, "open", _denied)
        with pytest.raises(SystemExit) as ei:
            _ensure_writable_output_dir(out)
        monkeypatch.undo()
        assert len(calls) == 1
        msg = str(ei.value.code)
        assert "cannot write to output directory" in msg
        assert "Permission denied" in msg
        assert '--user "$(id -u):$(id -g)"' in msg
        if hasattr(os, "getuid"):
            assert f"uid {os.getuid()}" in msg
        assert list(out.iterdir()) == []

    def test_undeletable_probe_warns_and_continues(self, tmp_path, monkeypatch, caplog):
        """A folder that accepts new files but refuses deletes can still take the ZIP,
        so only the create decides writability; the leftover probe is logged."""
        import logging
        import os
        from cli import _ensure_writable_output_dir

        def _no_delete(path, *a, **k):
            raise PermissionError(13, "Permission denied")

        out = tmp_path / "dropbox"
        out.mkdir()
        monkeypatch.setattr(os, "remove", _no_delete)
        with caplog.at_level(logging.WARNING, logger="cli"):
            _ensure_writable_output_dir(out)          # must not raise SystemExit
        monkeypatch.undo()
        left = list(out.iterdir())
        assert len(left) == 1 and left[0].name.startswith(".trace_write_test_")
        assert any("Could not remove the write-test file" in r.getMessage()
                   and str(left[0]) in r.getMessage() for r in caplog.records)

    @pytest.mark.skipif(not hasattr(__import__("os"), "getuid") or __import__("os").getuid() == 0,
                        reason="needs POSIX permissions and a non-root user")
    def test_readonly_existing_dir_exits_before_analysis(self, tmp_path, monkeypatch):
        import os
        from cli import run_pipeline
        self._no_analysis(monkeypatch)
        out = tmp_path / "ro"
        out.mkdir()
        os.chmod(out, 0o555)
        try:
            with pytest.raises(SystemExit) as ei:
                run_pipeline(self._args(tmp_path, out))
            assert "cannot write to output directory" in str(ei.value.code)
        finally:
            os.chmod(out, 0o755)

    def test_writable_dir_leaves_no_probe_behind(self, tmp_path):
        from cli import _ensure_writable_output_dir
        out = tmp_path / "a" / "b"
        _ensure_writable_output_dir(out)        # creates parents
        assert out.is_dir()
        assert list(out.iterdir()) == []         # probe removed
        _ensure_writable_output_dir(out)        # idempotent on an existing folder
        assert list(out.iterdir()) == []
