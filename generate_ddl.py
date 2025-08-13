import json

def generate_mysql_ddl(table_data):
    """Generates MySQL DDL from a JSON table definition."""
    table_name = table_data['table_name']
    table_comment = table_data.get('table_comment', '')
    cols = table_data['col']
    primary_keys = table_data['primary_keys']
    foreign_keys = table_data.get('foreign_keys', [])
    indexes = table_data.get('indexes', [])

    ddl_parts = []

    # Column definitions
    for col in cols:
        col_name = col['列名']
        col_type = col['列类型']
        is_nullable = col['是否可为空的']
        comment = col.get('注解', '')

        col_def = f"`{col_name}` {col_type}"
        if not is_nullable:
            col_def += " NOT NULL"
        if comment:
            col_def += f" COMMENT '{comment}'"
        ddl_parts.append(col_def)

    # Primary key
    if primary_keys:
        pk_cols = ", ".join([f"`{pk}`" for pk in primary_keys])
        ddl_parts.append(f"PRIMARY KEY ({pk_cols})")

    # Foreign keys
    for fk in foreign_keys:
        fk_col = fk['外键列名']
        ref_table = fk['引用表名']
        ref_col = fk['引用列名']
        fk_name = fk.get('约束名', f"fk_{table_name}_{fk_col}")
        ddl_parts.append(f"CONSTRAINT `{fk_name}` FOREIGN KEY (`{fk_col}`) REFERENCES `{ref_table}` (`{ref_col}`)")

    # Indexes
    for index in indexes:
        index_name = index['索引名']
        index_cols = index['列名']
        if not isinstance(index_cols, list):
            index_cols = [index_cols]
        idx_cols_str = ", ".join([f"`{c}`" for c in index_cols])
        ddl_parts.append(f"INDEX `{index_name}` ({idx_cols_str})")

    # Create table statement
    ddl = f"CREATE TABLE `{table_name}` (\n  "
    ddl += ",\n  ".join(ddl_parts)
    ddl += "\n)"

    # Table comment
    if table_comment:
        ddl += f" COMMENT='{table_comment}'"

    ddl += ";"

    return ddl

def generate_postgresql_ddl(table_data):
    """Generates PostgreSQL DDL from a JSON table definition."""
    table_name = table_data['table_name']
    table_comment_text = table_data.get('table_comment', '')
    cols = table_data['col']
    primary_keys = table_data['primary_keys']
    foreign_keys = table_data.get('foreign_keys', [])
    indexes = table_data.get('indexes', [])

    # Main CREATE TABLE statement
    ddl_parts = []
    comments = []

    if table_comment_text:
        comments.append(f'COMMENT ON TABLE "{table_name}" IS \'{table_comment_text}\';')

    # Column definitions
    for col in cols:
        col_name = col['列名']
        col_type = col['列类型']
        is_nullable = col['是否可为空的']
        comment = col.get('注解', '')

        col_def = f'"{col_name}" {col_type}'
        if not is_nullable:
            col_def += " NOT NULL"
        ddl_parts.append(col_def)

        if comment:
            comments.append(f'COMMENT ON COLUMN "{table_name}"."{col_name}" IS \'{comment}\';')

    # Primary key
    if primary_keys:
        pk_cols = ", ".join([f'"{pk}"' for pk in primary_keys])
        ddl_parts.append(f"PRIMARY KEY ({pk_cols})")

    # Foreign keys
    for fk in foreign_keys:
        fk_col = fk['外键列名']
        ref_table = fk['引用表名']
        ref_col = fk['引用列名']
        fk_name = fk.get('约束名', f"fk_{table_name}_{fk_col}")
        ddl_parts.append(f'CONSTRAINT "{fk_name}" FOREIGN KEY ("{fk_col}") REFERENCES "{ref_table}" ("{ref_col}")')

    ddl = f'CREATE TABLE "{table_name}" (\n  '
    ddl += ",\n  ".join(ddl_parts)
    ddl += "\n);"

    # Comments
    if comments:
        ddl += "\n\n" + "\n".join(comments)

    # Indexes (as separate statements)
    if indexes:
        ddl += "\n"
        for index in indexes:
            index_name = index['索引名']
            index_cols = index['列名']
            if not isinstance(index_cols, list):
                index_cols = [index_cols]
            idx_cols_str = ", ".join([f'"{c}"' for c in index_cols])
            ddl += f'\nCREATE INDEX "{index_name}" ON "{table_name}" ({idx_cols_str});'

    return ddl

def main():
    # The JSON data provided by the user, now including table_comment.
    # Note: I have corrected the missing comma before "constraints".
    json_data = """
    {
        "table_name": "user_profiles",
        "table_comment": "用户个人资料表 (User Profile Table)",
        "col": [
            {"列名": "id", "列类型": "int", "是否可为空的": false, "注解": "Primary key identifier"},
            {"列名": "user_id", "列类型": "int", "是否可为空的": false, "注解": "Foreign key to users table"},
            {"列名": "bio", "列类型": "text", "是否可为空的": true, "注解": "User biography"},
            {"列名": "location", "列类型": "varchar(255)", "是否可为空的": true, "注解": null}
        ],
        "primary_keys": ["id"],
        "foreign_keys": [
            {"外键列名": "user_id", "引用表名": "users", "引用列名": "id"}
        ],
        "indexes": [
            {"索引名": "idx_location", "列名": "location"}
        ],
        "constraints": [
            {"约束名": "pk_user_profiles", "约束类型": "主键"}
        ]
    }
    """

    table_data = json.loads(json_data)

    print("--- MySQL DDL ---")
    mysql_ddl = generate_mysql_ddl(table_data)
    print(mysql_ddl)

    print("\n--- PostgreSQL DDL ---")
    postgres_ddl = generate_postgresql_ddl(table_data)
    print(postgres_ddl)

if __name__ == "__main__":
    main()
