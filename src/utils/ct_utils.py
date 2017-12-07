pipe.remove_large_negative_values_from_ct(
    self.ct_scan, subjects_dir=self.subjects_dir,
    subject=self.subject)

aff = pipe.register_ct_to_mr_using_mutual_information(
                self.ct_scan, subjects_dir=self.subjects_dir,
                subject=self.subject, overwrite=overwrite,
                registration_algorithm=['--cost', 'nmi'])