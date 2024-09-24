from typing import Sequence

import numpy as np
import pandas as pd
import logging
from sqlalchemy import create_engine, Column, Integer, Float, String, select, Table, MetaData
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()
logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self):
        pass

    def load(self):
        pass


class DataModel(Base):
    """
    SQLAlchemy ORM class for the measurements table.
    """
    __tablename__ = 'measurements'
    step = Column(Integer, primary_key=True)
    bus = Column(Integer, primary_key=True)
    freq = Column(Float, primary_key=True)
    element = Column(String, primary_key=True)
    element_type = Column(Integer, primary_key=True)
    v1 = Column(Float)
    vangle1 = Column(Float)
    v2 = Column(Float)
    vangle2 = Column(Float)
    v3 = Column(Float)
    vangle3 = Column(Float)
    i1 = Column(Float)
    iangle1 = Column(Float)
    i2 = Column(Float)
    iangle2 = Column(Float)
    i3 = Column(Float)
    iangle3 = Column(Float)
    s1 = Column(Float)
    ang1 = Column(Float)
    s2 = Column(Float)
    ang2 = Column(Float)
    s3 = Column(Float)
    ang3 = Column(Float)


def _get_injected_currents(df):
    """
    Calculate injected currents for each harmonic at each bus and join with the voltages where element_type is 0.

    :param df: DataFrame with all columns, indexed by step, bus, element, element_type, and freq.
               Currents and voltages are represented by v1 to v3, i1 to i3 and their respective angles by iangle1 to iangle3.
    :return: DataFrame with step, freq, bus as index, containing injected currents and voltages.
    """
    # Extract and prepare voltage data for merging

    voltage_data = df[df.index.get_level_values('element_type') == 0]
    voltage_cols = [col for col in df.columns if 'v' in col[:2]]  # Assumes voltage columns are named like v1, v2, v3
    voltage_data = voltage_data[voltage_cols].drop_duplicates()
    voltage_data = voltage_data.reset_index().drop(['element', 'element_type'], axis=1)

    # Prepare current data with complex calculations
    phase_cols = [f'i{i}' for i in range(1, 4) if f'i{i}' in df.columns]
    phases = [i for i in range(1, 4) if f'i{i}' in df.columns]
    angle_cols = [f'iangle{i}' for i in range(1, 4) if f'iangle{i}' in df.columns]

    relevant_types = df[(df.index.get_level_values('element_type') == 1) |
                        (df.index.get_level_values('element_type') == 2) |
                        (df.index.get_level_values('element_type') == 3)]

    if not relevant_types.empty:
        for i, angle in zip(phase_cols, angle_cols):
            relevant_types[f'complex_{i}'] = relevant_types[i] * np.exp(1j * np.radians(relevant_types[angle]))
            mask = relevant_types.index.get_level_values('element_type') == 1
            relevant_types.loc[mask, f'complex_{i}'] *= -1

        complex_aggregates = {f'complex_{i}': 'sum' for i in phase_cols}
        grouped = relevant_types.groupby(['step', 'freq', 'bus']).agg(complex_aggregates)

        for i in phase_cols:
            grouped[i] = np.abs(grouped[f'complex_{i}'])
            grouped[f'{i.replace("i", "iangle")}'] = np.degrees(np.angle(grouped[f'complex_{i}']))
            del grouped[f'complex_{i}']

        # Reset index to merge
        grouped = grouped.reset_index()
    else:
        # Create an empty DataFrame with necessary columns if no relevant types found
        grouped = pd.DataFrame(columns=['step', 'freq', 'bus'] + phase_cols + [f'iangle{i}' for i in phases])

    # Merge current data with voltage data
    result_df = pd.merge(grouped, voltage_data, on=['step', 'freq', 'bus'], how='right')

    # Fill NaN values for currents and angles with zeros
    for col in phase_cols + [f'iangle{i}' for i in phases]:
        result_df[col].fillna(0, inplace=True)

    return result_df.set_index(['step', 'freq', 'bus'])


class DataLoaderSQL(DataLoader):
    def __init__(self, url):
        """
        Initialize the DataLoader with database configuration.

        :param url: A database URL for SQLAlchemy connection
        :param table_name: Table name to query from
        """
        super().__init__()
        self.engine = create_engine(url)
        self.Session = sessionmaker(bind=self.engine)
        self.metadata = MetaData()

    def load(self, table_name=None, steps=None, busses=None, frequencies=None, elements=None,
             element_types=None, columns=None):
        session = self.Session()
        table = Table(table_name, self.metadata, autoload_with=self.engine)

        default_columns = ['step', 'bus', 'element', 'element_type', 'freq']
        if columns is not None:
            all_columns = set(columns + default_columns)
        else:
            all_columns = set([col.name for col in table.columns])
        # Construct the dynamic query
        if columns:
            query = select(*[table.c[col] for col in all_columns if col in table.c])
        else:
            query = select(table)

        # Apply filters based on the function parameters
        if steps:
            query = query.filter(table.c.step.in_(steps))
        if busses:
            query = query.filter(table.c.bus.in_(busses))
        if frequencies:
            query = query.filter(table.c.freq.in_(frequencies))
        if elements:
            query = query.filter(table.c.element.in_(elements))
        if element_types:
            query = query.filter(table.c.element_type.in_(element_types))

        # Execute the query and fetch results
        result = session.execute(query).fetchall()
        session.close()

        df = pd.DataFrame(result, columns=[col.key for col in query.selected_columns])
        # Set DataFrame index if all key columns are present
        if set(default_columns).issubset(df.columns):
            df.set_index(default_columns, inplace=True)

        df = _get_injected_currents(df)
        return df
