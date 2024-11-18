import pandas as pd
if __name__ == '__main__':
    path = '../result/'
    case_union = pd.read_csv(path + '/case_union_m.csv', index_col=0, header=0, sep=",")
    control_union = pd.read_csv(path + '/control_union_m.csv', index_col=0, header=0, sep=",")



    # Get the number of nodes
    num_nodes_1 = case_union.shape[0]
    nodes_1 = list(range(num_nodes_1))

    # Create an empty DataFrame
    result_case = pd.DataFrame(columns=['Node', 'AdjacentNode', 'Correlation'])

    # Traverse the adjacency matrix
    for i in nodes_1:
        for j in nodes_1:
            correlation_1 = case_union.iloc[i, j]
            if not pd.isna(correlation_1) and correlation_1 != 0:
                # Extract correlation value
                corr_sign = -1 if correlation_1 < 0 else 1

                # Add node and correlation information to the DataFrame
                result_case = result_case.append({
                    'Node': i,
                    'AdjacentNode': j,
                    'Correlation':correlation_1
                }, ignore_index=True)

    num_nodes_2 = control_union.shape[0]
    nodes = list(range(num_nodes_2))

    # Create an empty DataFrame
    result_control = pd.DataFrame(columns=['Node', 'AdjacentNode', 'Correlation'])

    # Traverse the adjacency matrix
    for i in nodes:
        for j in nodes:
            correlation_2 = control_union.iloc[i, j]
            if not pd.isna(correlation_2) and correlation_2 != 0:
                # Extract correlation value
                corr_sign = -1 if correlation_2 < 0 else 1

                # Add node and correlation information to the DataFrame
                result_control = result_control.append({
                    'Node': i,
                    'AdjacentNode': j,
                    'Correlation':correlation_2
                }, ignore_index=True)
    # Drop rows where Node is equal to AdjacentNode

    result_case = result_case[result_case['Node'] != result_case['AdjacentNode']]
    result_case = result_case[result_case['Node'] < result_case['AdjacentNode']]
    result_case = result_case.drop_duplicates(subset=['Node', 'AdjacentNode']).reset_index(drop=True)
    result_case.to_csv(path +'/gcase_m.csv', index=False)
    result_control = result_control[result_control['Node'] != result_control['AdjacentNode']]
    result_control = result_control[result_control['Node'] < result_control['AdjacentNode']]
    result_control = result_control.drop_duplicates(subset=['Node', 'AdjacentNode']).reset_index(drop=True)
    result_control.to_csv(path +'/gcontrol_m.csv', index=False)