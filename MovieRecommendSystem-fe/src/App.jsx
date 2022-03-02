import { useState, useEffect } from 'react'
import { Table, Button, Select, Form, Input } from 'antd';
import axios from 'axios';
import 'antd/dist/antd.css'
import './App.css'
import MyImage from './components/MyImage';
import MyLayout from './components/MyLayout';

function App() {

  const dataSource = [
    {
      title: '阿凡达',
      grade: 4.8,
      recommender: 32,
      genres: ["科幻", "战争", "爱情"]
    },
    {
      title: '头号玩家',
      grade: 4.3, 
      recommender:14,
      genres: ["科幻", "猎奇"]
    },
  ];
  const [data, setData] = useState(dataSource);

  const { Option } = Select;

  function handleChange(value) {
    console.log(`selected ${value}`);
  }

  useEffect(() => {
    // const res = fetch();
    // setData(res.data);
  }, []);

  const columns = [
    {
      title: 'Title',
      dataIndex: 'Title',
      key: 'Title',
    },
    {
      title: 'Grade',
      dataIndex: 'Grade',
      key: 'Grade',
    },
    {
      title: 'Recommender',
      dataIndex: 'Recommender',
      key: 'Recommender',
    },
    {
      title: 'Genres',
      dataIndex: 'Genres',
      key: 'Genres'
    }
  ];

  function load() {
    setData([]);
    axios
      .get('http://127.0.0.1:5000/recommend')
      .then(res => {
        setData(res.data)
        console.log(res)
      });
  }

  const onFinish = values => {
    // console.log(values);
    // embed_value = values['embed'];
    // recommend_value = values['recommend'];
    // uid = values.uid;
    axios
      .get('http://127.0.0.1:5000/recommend', {
        params : {
          embed : values['embed'],
          recommend : values['recommend'],
          uid : values['uid']
        }
      }).then(res => {
        setData(res.data)
        console.log(res)
      });
    console.log('Success:', values);
  };

  const onFinishFailed = errorInfo => {
    console.log('Failed:', errorInfo);
  };

  return (
    <div className="App">
      <MyLayout>
        {/* <Button onClick={handleClick}>click</Button> */}
        <div className="form-container">
          <Form
            className="form"
            name="basic"
            labelCol={{ span: 8 }}
            wrapperCol={{ span: 12 }}
            initialValues={{ remember: true }}
            onFinish={onFinish}
            onFinishFailed={onFinishFailed}
            autoComplete="off"
          >
            <Form.Item
              label="UID"
              name="uid"
              rules={[{ required: true, message: 'require' }]}
            >
              <Input />
            </Form.Item>
            <Form.Item
              label="嵌入算法"
              name="embed"
              rules={[{ required: true, message: 'require' }]}
            >
              <Select>
                <Option value="trans-e" >TransE</Option>
                <Option value="trans-h" >TransH</Option>
                {/* <Option value="TransR（未实现）" >X</Option> */}
              </Select>
            </Form.Item>
            <Form.Item
              label="推荐算法"
              name="recommend"
              rules={[{ required: true, message: 'require' }]}
            >
              <Select>
                <Option value="user-cf" >User-CF</Option>
                <Option value="item-cf" >Item-CF</Option>
              </Select>
            </Form.Item>
            <Form.Item wrapperCol={{ offset: 2, span: 16 }}>
              <Button type="primary" htmlType="submit">
                获取推荐结果
              </Button>
            </Form.Item>
          </Form>
        </div>
        <div className="table-container">
          <Table dataSource={data} columns={columns} />
          <Button type="primary" onClick={load}>Load</Button>
        </div>
      </MyLayout>
    </div>
  )
}

export default App
