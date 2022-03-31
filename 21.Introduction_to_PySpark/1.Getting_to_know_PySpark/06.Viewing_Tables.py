# Print the tables in the catalog
print(spark.catalog.listTables())

# output:[Table(name='flights', database=None, description=None, tableType='TEMPORARY', isTemporary=True)]

# ======================================================================== #