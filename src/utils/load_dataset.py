"""Loading and querying a longitudinal subject/session dataset (CSV or DataFrame)."""

from __future__ import annotations

import numpy as np
import pandas as pd


class LoadDataset:
    """Wraps a longitudinal subject-session table with convenience queries.

    The underlying data is expected to have one row per (subject, session)
    with at least a subject-id column, a session-id column, and an age
    column (column names are configurable via the constructor).
    """

    def __init__(
        self,
        data: str | pd.DataFrame | None = None,
        sid_column: str = "id",
        session_column: str = "session_id",
        age_column: str = "age",
    ) -> None:
        """Load the dataset from a CSV path or an existing DataFrame.

        Args:
            data: Either a path to a CSV file, or an already-loaded
                ``pandas.DataFrame``.
            sid_column: Name of the subject-id column.
            session_column: Name of the session-id column (expected to
                follow a ``"m<months>"`` naming convention, e.g.
                ``"m000"``, ``"m024"`` — see
                :meth:`get_ids_with_followup_gap`).
            age_column: Name of the (baseline or per-session) age column.

        Raises:
            ValueError: If ``data`` is neither a string nor a DataFrame.
        """
        self.sid_column = sid_column
        self.session_column = session_column
        self.age_column = age_column

        if isinstance(data, str):
            self.data_path_name = data
            self.df = pd.read_csv(self.data_path_name)
        elif isinstance(data, pd.DataFrame):
            self.df = data
        else:
            raise ValueError("data must be a string or a pandas DataFrame")

    def set_df(self, df: pd.DataFrame) -> None:
        """Replace the underlying DataFrame."""
        self.df = df

    def get_df(self) -> pd.DataFrame:
        """Return the underlying DataFrame."""
        return self.df

    def get_timepoint_df(self, timepoint: str = "m000") -> pd.DataFrame:
        """Return only the rows for a given session/timepoint.

        Args:
            timepoint: Session-id value to filter on (e.g. ``"m000"`` for
                baseline).

        Returns:
            Subset of ``self.df`` where ``session_column == timepoint``.
        """
        return self.df[self.df[self.session_column] == timepoint]

    def get_baseline_df(self, index: int = 0) -> pd.DataFrame:
        """Return one row per subject, selecting a specific visit by order.

        Rows are first sorted by ``(subject_id, session_id)``, then grouped
        by subject.

        Args:
            index: Which visit to select per subject:
                - ``0``: the first (earliest) visit — i.e. the baseline.
                - ``-1``: the last (most recent) visit.
                - any other integer: the visit at that position (via
                  ``groupby(...).nth(index)``), which may result in fewer
                  than one row for subjects without a visit at that index.

        Returns:
            One-row-per-subject DataFrame (except when ``index`` selects a
            visit that some subjects don't have).
        """
        ordered_df = self.df.sort_values(by=[self.sid_column, self.session_column])

        if index == 0:
            return ordered_df.groupby(self.sid_column).first().reset_index()
        if index == -1:
            return ordered_df.groupby(self.sid_column).last().reset_index()
        return ordered_df.groupby(self.sid_column).nth(index).reset_index()

    def get_ids_with_followup(self) -> np.ndarray:
        """Return sorted subject ids that have more than one session.

        Returns:
            Sorted array of subject ids appearing more than once in
            ``self.df``.
        """
        session_counts = self.df[self.sid_column].value_counts()
        subject_ids = session_counts[session_counts > 1].index.values
        return np.sort(subject_ids)

    def get_ids_with_no_followup(self) -> np.ndarray:
        """Return sorted subject ids that have exactly one session.

        Returns:
            Sorted array of subject ids appearing exactly once in
            ``self.df``.
        """
        session_counts = self.df[self.sid_column].value_counts()
        subject_ids = session_counts[session_counts == 1].index
        return np.sort(subject_ids)

    def get_ids_with_followup_gap(
        self,
        months: int | tuple[int, int] | list[int],
        to_baseline_only: bool = False,
    ) -> list[tuple[object, object, object, float]]:
        """Find all timepoint pairs, per subject, matching a follow-up gap.

        Session ids are assumed to encode months since baseline as
        ``"m<months>"`` (e.g. ``"m024"`` -> 24 months); only the numeric
        suffix (``session_id[1:]``) is parsed.

        Args:
            months: Either a single integer (exact gap match, in months)
                or a 2-element ``(min, max)`` range (exclusive bounds:
                ``min < gap < max``).
            to_baseline_only: If True, for each subject only pairs
                involving that subject's earliest available session are
                considered (i.e. the outer timepoint-pair loop stops after
                its first iteration — see implementation note below).

        Returns:
            A list of ``(subject_id, first_session_id, second_session_id,
            baseline_age)`` tuples, one per matching timepoint pair, where
            ``baseline_age`` is the subject's age at ``first_session_id``.
        """
        if not isinstance(months, (list, tuple)):
            months = [months]
        else:
            months = list(months)
        months = [int(month) for month in months]

        followup_ids = self.get_ids_with_followup()
        ordered_df = self.df.sort_values(by=[self.sid_column, self.session_column])

        matching_pairs = []
        for subject_id in followup_ids:
            subject_timepoints = ordered_df[ordered_df[self.sid_column] == subject_id]
            subject_months = np.array([int(session[1:]) for session in subject_timepoints[self.session_column].values])

            for i in range(len(subject_months)):
                for j in range(i + 1, len(subject_months)):
                    gap = subject_months[j] - subject_months[i]
                    exact_match = len(months) == 1 and gap == months
                    range_match = len(months) == 2 and months[0] < gap < months[1]
                    if exact_match or range_match:
                        matching_pairs.append(
                            (
                                subject_timepoints.iloc[0][self.sid_column],
                                subject_timepoints.iloc[i][self.session_column],
                                subject_timepoints.iloc[j][self.session_column],
                                float(subject_timepoints.iloc[i][self.age_column]),
                            )
                        )
                # NOTE: `break` is intentionally here, outside the inner `j`
                # loop: when `to_baseline_only=True`, this stops the outer
                # `i` loop after i=0 (the subject's earliest timepoint),
                # so only pairs anchored at the baseline visit are kept.
                if to_baseline_only:
                    break
        return matching_pairs

    def filter_by_follow_up_gap(
        self,
        months: int | tuple[int, int] | list[int],
        to_baseline_only: bool = False,
    ) -> pd.DataFrame:
        """Build a DataFrame of baseline+follow-up row pairs matching a gap.

        Args:
            months: See :meth:`get_ids_with_followup_gap`.
            to_baseline_only: See :meth:`get_ids_with_followup_gap`.

        Returns:
            A DataFrame with the same columns as ``self.df``, containing
            two consecutive rows (first session, then second session) for
            every matching pair found by :meth:`get_ids_with_followup_gap`.
        """
        matching_pairs = self.get_ids_with_followup_gap(months, to_baseline_only=to_baseline_only)
        matched_rows = []
        for subject_id, first_session_id, second_session_id, _baseline_age in matching_pairs:
            matched_rows.append(
                self.df[(self.df[self.sid_column] == subject_id) & (self.df[self.session_column] == first_session_id)].values[0]
            )
            matched_rows.append(
                self.df[(self.df[self.sid_column] == subject_id) & (self.df[self.session_column] == second_session_id)].values[0]
            )

        return pd.DataFrame(matched_rows, columns=self.df.columns)

    def get_sessions_list(self, ascending: bool = True, include_ages: bool = False) -> dict:
        """Build a mapping from subject id to their ordered list of sessions.

        Args:
            ascending: Sort order for sessions within each subject.
            include_ages: If True, each session entry is a
                ``(session_id, age)`` tuple instead of just ``session_id``.

        Returns:
            Dictionary mapping subject id -> list of session ids (or
            ``(session_id, age)`` tuples if ``include_ages=True``), in the
            requested order.
        """
        ordered_df = self.df.sort_values(by=[self.sid_column, self.session_column], ascending=ascending)

        sessions_by_subject: dict = {}
        current_subject_id = None
        for _, row in ordered_df.iterrows():
            session_entry = (row[self.session_column], row[self.age_column]) if include_ages else row[self.session_column]
            if current_subject_id != row[self.sid_column]:
                current_subject_id = row[self.sid_column]
                sessions_by_subject[current_subject_id] = [session_entry]
            else:
                sessions_by_subject[current_subject_id].append(session_entry)
        return sessions_by_subject

    def get_subject_session_row(self, s_id: object, session_id: object) -> pd.Series | None:
        """Return the row for a specific (subject, session) pair.

        Args:
            s_id: Subject id to look up.
            session_id: Session id to look up.

        Returns:
            The matching row as a ``pandas.Series``, or ``None`` if no
            such (subject, session) pair exists.
        """
        matching_rows = self.df[(self.df[self.sid_column] == s_id) & (self.df[self.session_column] == session_id)]
        if matching_rows.empty:
            return None
        return matching_rows.iloc[0]