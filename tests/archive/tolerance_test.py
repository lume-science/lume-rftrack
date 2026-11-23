"""archive()/load_archive() leniency toward RF-Track changes: a value HDF5
can't hold is dropped, `strict=False` archives an element that doesn't
rebuild with the same field, and loading warns about either, and about an
archive another RF-Track wrote.
"""


def test_loaded_with_another_rftrack_warns():
    import h5py
    import RF_Track as rft
    import rftrack_util
    from pykern import pkunit

    from lume_rftrack.simulator import RFTrackSimulator

    path = str(pkunit.empty_work_dir().join("a.h5"))
    RFTrackSimulator(
        _lattice(rft, rft.Drift(0.1)), rftrack_util.twiss_bunch(rft, 100.0, 10)
    ).archive(path)
    pkunit.pkeq([], _load_warnings(path))
    with h5py.File(path, "r+") as f:
        f.attrs["rftrack_version"] = "0.0.0"
    w = _load_warnings(path)
    pkunit.pkeq(1, len(w), "warnings={}", w)
    pkunit.pkok("RF-Track 0.0.0" in w[0], "warning={}", w[0])


def test_strict_false_archives_a_different_field():
    """A Coil built without its constructor arguments has no field outside
    its length (see `_constructor_args_coil()`), so archiving it without
    them rebuilds a different field."""
    import warnings

    import RF_Track as rft
    import rftrack_util
    from pykern import pkunit

    import lume_rftrack.archive_elements
    from lume_rftrack.archive import read_archive
    from lume_rftrack.simulator import RFTrackSimulator

    path = str(pkunit.empty_work_dir().join("a.h5"))
    s = RFTrackSimulator(
        _lattice(rft, rft.Coil(0.2, 0.1, 0.05)),
        rftrack_util.twiss_bunch(rft, 100.0, 10),
    )
    f = lume_rftrack.archive_elements._constructor_args_coil
    del lume_rftrack.archive_elements._constructor_args_coil
    try:
        with pkunit.pkexcept("strict=False"):
            s.archive(path)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            s.archive(path, strict=False)
    finally:
        lume_rftrack.archive_elements._constructor_args_coil = f
    w = [str(x.message) for x in w if issubclass(x.category, UserWarning)]
    pkunit.pkeq(1, len(w), "warnings={}", w)
    pkunit.pkok("element 0 'E0'" in w[0], "warning={}", w[0])
    e = read_archive(path).elements[0]
    pkunit.pkok(e.restore_field_error > 0.01, "restore_field_error={}", e)
    w = _load_warnings(path)
    pkunit.pkeq(1, len(w), "warnings={}", w)
    pkunit.pkok("strict=False" in w[0], "warning={}", w[0])


def test_unstorable_attribute_is_dropped():
    """e.g. RF-Track 2.8.0's `get_position()`/`set_position()`, a Frame,
    had it not been excluded by name"""
    import RF_Track as rft
    from pykern import pkunit

    from lume_rftrack.archive_elements import to_dict

    # named Drift: archive rebuilds an element by its class name
    class Drift(rft.Drift):
        def get_widget(self):
            return object()

        def set_widget(self, value):
            pass

    d = to_dict(rft, Drift(0.1))
    pkunit.pkeq(["widget"], d.unarchived_attributes)
    pkunit.pkok("widget" not in d.attributes, "attributes={}", d.attributes)
    pkunit.pkok("length" in d.attributes, "attributes={}", d.attributes)


def _lattice(rft, element):
    rv = rft.Lattice()
    element.set_name("E0")
    rv.append(element)
    return rv


def _load_warnings(path):
    import warnings

    from lume_rftrack.simulator import RFTrackSimulator

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        RFTrackSimulator().load_archive(path)
    return [str(x.message) for x in w if issubclass(x.category, UserWarning)]
