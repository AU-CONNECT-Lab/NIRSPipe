"""What both report paths read: the page shell, figure IO, and the shared tables.

The bottom of the three. Nothing here may import :mod:`fnirs_pipe.qc.subject` or
:mod:`fnirs_pipe.qc.hyper`; a helper that needs one of them belongs in that one.

This layer exists because of a bug it would have prevented. ``condition_windows`` and
``markers_on_data_axis`` lived in the dyad report, the subject reports imported them from
there, and when the ``first_time`` correction was written it was applied on the path whose
module held the code and not on the other. A fix follows the module it is written in, not
the meaning it belongs to, so shared meaning needs a module of its own.
"""
