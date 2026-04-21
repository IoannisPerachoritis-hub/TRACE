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
        assert args.n_pcs == 4
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
            "--no-report", "--no-plots", "--drop-alt", "-v",
        ])
        assert args.no_report is True
        assert args.no_plots is True
        assert args.drop_alt is True
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

    def test_auto_pcs_defaults(self):
        parser = _build_parser()
        args = parser.parse_args([
            "--vcf", "x.vcf", "--pheno", "p.csv", "--trait", "Y", "--output", "o/",
        ])
        assert args.auto_pcs is False
        assert args.pc_strategy == "band"
        assert args.max_pcs == 10

    def test_auto_pcs_enabled(self):
        parser = _build_parser()
        args = parser.parse_args([
            "--vcf", "x.vcf", "--pheno", "p.csv", "--trait", "Y", "--output", "o/",
            "--auto-pcs", "--pc-strategy", "closest_to_1", "--max-pcs", "15",
        ])
        assert args.auto_pcs is True
        assert args.pc_strategy == "closest_to_1"
        assert args.max_pcs == 15

    def test_invalid_pc_strategy_rejected(self):
        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([
                "--vcf", "x.vcf", "--pheno", "p.csv", "--trait", "Y", "--output", "o/",
                "--pc-strategy", "invalid",
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
        assert args.sig_thresh == "meff"

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
        args.auto_pcs = True
        args.pc_strategy = "closest_to_1"
        args.subsampling = True
        args.boot_reps = 50
        args.boot_jobs = -1
        cmd = _build_equivalent_command(args)
        assert "--model mlm mlmm farmcpu" in cmd
        assert "--auto-pcs" in cmd
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
        # default meff — should NOT appear
        args.sig_thresh = "meff"
        cmd = _build_equivalent_command(args)
        assert "--sig-thresh" not in cmd
        # non-default — should appear
        args.sig_thresh = "bonferroni"
        cmd = _build_equivalent_command(args)
        assert "--sig-thresh bonferroni" in cmd

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
