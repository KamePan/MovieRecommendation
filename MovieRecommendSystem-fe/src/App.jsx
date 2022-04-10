import { useState, useEffect } from 'react'
import { Routes, Route } from 'react-router-dom';
import { Table, Button, Select, Form, Input } from 'antd';
import axios from 'axios';
import 'antd/dist/antd.css'
import './App.css'
import Header from './components/Header';
import Person from './pages/Person';
import Recommend from './pages/Recommed';

const defaultMovie = {
  title: '你的名字。',
  rate: 9.8,
  showtime: '2021',
  length: 120,
  district: ['中国', '日本'],
  language: '日语',
  actor: ['潘恺铭', '宫水三叶'],
  director: ['新海诚'],
  othername: [],
  doubanUrl: 'https://movie.douban.com/subject/26683290/',
  coverUrl: 'src/assets/p2395733377.webp',
};

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
    recommender: 14,
    genres: ["科幻", "猎奇"]
  },
];

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

function App() {

  // const [data, setData] = useState(dataSource);
  // const [movieInfo, setMovieInfo] = useState(defaultMovie);

  // const { Option } = Select;

  // function handleChange(value) {
  //   console.log(`selected ${value}`);
  // }

  // useEffect(() => {
  //   // const res = fetch();
  //   // setData(res.data);
  // }, []);


  // function load() {
  //   setData([]);
  //   axios
  //     .get('http://127.0.0.1:5000/recommend')
  //     .then(res => {
  //       setData(res.data)
  //       console.log(res)
  //     });
  // }

  // const onFinish = values => {
  //   // console.log(values);
  //   // embed_value = values['embed'];
  //   // recommend_value = values['recommend'];
  //   // uid = values.uid;
  //   axios
  //     .get('http://127.0.0.1:5000/recommend', {
  //       params: {
  //         embed: values['embed'],
  //         recommend: values['recommend'],
  //         uid: values['uid']
  //       }
  //     }).then(res => {
  //       setData(res.data)
  //       console.log(res)
  //     });
  //   console.log('Success:', values);
  // };

  // const onFinishFailed = errorInfo => {
  //   console.log('Failed:', errorInfo);
  // };

  return (
    <div className="min-h-screen">
      <Header />
      <Routes>
        <Route path="/" element={<Recommend />}></Route>
        <Route path="/person" element={<Person />}></Route>
      </Routes>
    </div>
  )
}

export default App
